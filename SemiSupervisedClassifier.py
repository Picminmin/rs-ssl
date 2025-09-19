import sys, os
import numpy as np
from itertools import chain
from sklearn.base import BaseEstimator, ClassifierMixin
from types import SimpleNamespace
from sklearn.metrics import accuracy_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class SemiSupervisedClassifier(BaseEstimator, ClassifierMixin):
    """
    ラベル付きデータから見て空間的に隣接したラベルなしデータのラベル拡張を伴う
    (ラベル拡張はすべてのラベルなしデータに行うのが良いとは限らない)半教師付き分類の雛形

    - mode="confidence": 信頼度ベース
    - mode="spatial": 空間的隣接性ベース(upd_LUlabelを使用)
    - mode="hybrid": 信頼度と空間的隣接性を組み合わせ
    """

    def __init__(self, base_clf, mode="confidence",
                 conf_threshold=0.9, max_iter = 10,
                 random_state = None):
        self.base_clf = base_clf
        self.mode = mode
        self.conf_threshold = conf_threshold
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, L_index, U_index, X, y, image_shape, upd_LUlabel = None):
        """
        Args:
            L_index (ndarray): ラベル付きデータのインデックス
            U_index (ndarray): ラベルなしデータのインデックス
            X (ndarray): 特徴量 (n_samples, n_features)
            y (ndarray): 正解ラベル (n_samples,)
            image_shape (tuple): (H, W)
            upd_LUlabel (function, optional): 空間拡張の関数 (mode="spatial"/"hybrid"で必須)
        """
        H, W = image_shape
        rng = np.random.RandomState(self.random_state)

        # 拡張ラベルを保持する配列
        expand_label = np.zeros_like(y, dtype=np.int32)
        expand_label[L_index] = y[L_index]

        for it in range(self.max_iter):
            print(f"[INFO] Iter {it+1}/{self.max_iter}, "
                  f"L={len(L_index)}, U={len(U_index)}")

            if len(U_index) == 0:
                break

            # --- 学習 ---
            self.base_clf.fit(X[L_index], expand_label[L_index])

            # --- 予測 ---
            probs = None
            if hasattr(self.base_clf, "predict_proba"):
                probs = self.base_clf.predict_proba(X[U_index])
                y_pred = probs.argmax(axis = 1)
                conf = probs.max(axis = 1)

            else:
                y_pred = self.base_clf.predict(X[U_index])
                conf = np.ones_like(y_pred, dtype = float) # 信頼度情報なし

            # --- 拡張 ---
            if self.mode == "confidence":
                mask = conf >= self.conf_threshold
                if not np.any(mask):
                    print(f"[INFO] No confident pseudo-labels → stop")
                    break

                new_idx = U_index[mask]
                expand_label[new_idx] = y_pred[mask]
                L_index = np.concatenate([L_index, new_idx])
                U_index = U_index[~mask]

            elif self.mode in ["spatial", "hybrid"]:
                if upd_LUlabel is None:
                    raise ValueError("upd_LUlabel 関数を渡してください")

                # 信頼度でしぼる (hybridモードのみ)
                if self.mode == "hybrid":
                    candidate_idx = U_index[conf >= self.conf_threshold]
                else:
                    candidate_idx = U_index

                L_index, U_index, upd_flag, expand_label = upd_LUlabel(
                    pre_L = L_index,
                    pre_U = U_index,
                    y_pred = y_pred, # conf使いたければここに渡せる
                    expand_label = expand_label,
                    ground_truth = y.reshape(image_shape), # 2Dで渡すと便利
                    background_label = 0,
                    connectivity = 8
                )

                if not upd_flag:
                    print("[INFO] No spatial expansion → stop")
                    break


if __name__ == "__main__":
    from sklearn.svm import SVC
    from dataclasses import dataclass
    from .utils.spatial_expansion import upd_LUlabel # rs-sslをパッケージとして扱うため、utilsの先頭にドットを付けて相対指定する
    from pprint import pprint
    from RS_GroundTruth.rs_dataset import RemoteSensingDataset  # あなたのrs_dataset.py
    from spatialcv.spatial_train_test_split import spatial_train_test_split, SpatialSplitConfig

    pprint(sys.path[0])
    # データ読み込み
    ds = RemoteSensingDataset(remove_bad_bands=True)
    X, y = ds.load("Indianpines") # X.shape = (H, W, B), y.shape= (H, W)
    H, W, B = X.shape
    X_flat = X.reshape(-1, B)
    y_flat = y.flatten()

    # --- train/test分割 ---
    split_random_state = 43
    # テストサイズの指定
    test_size, error_rate = 0.6, 0.1
    class SpatialSplitConfig:
        n_rows: int = 13                       # ブロック分割数(縦)
        n_cols: int = 13                       # ブロック分割数(横)
        min_train_samples_per_class: int = 5  # 各土地被覆クラスで作成する教師データのサンプル数の下限
        min_test_samples_per_class: int = 5    # 各土地被覆クラスで作成するなテストサンプル数の下限
        background_label: int = 0              # 各土地被覆クラスで作成するなテストサンプル数の下限
        random_state: int = split_random_state       # 乱数シード
        auto_adjust_test_size: bool = True    # テストサイズを自動調整するか
        min_search_test_ratio: float = test_size - error_rate    # 自動探索モード時の下限
        max_search_test_ratio: float = test_size + error_rate    # 自動探索モード時の上限
        step: float = 0.05                     # 自動探索モード時の刻み幅
        max_iter: int = 100                    # ランダム試行回数

    cfg = SpatialSplitConfig()
    X_train, X_test, y_train, y_test, train_mask, test_mask, best_ts = spatial_train_test_split(
        X = X, y = y, test_size = test_size, cfg = cfg
    )


    # --- インデックス作成 ---
    train_index = np.where(train_mask.flatten())[0]
    test_index = np.where(test_mask.flatten())[0]

    L_index = train_index # ラベル付き = train_mask
    # U_indexは別途調整が必要である。本来は教師を除くすべてのインデックスをU_indexに指定する必要がある。
    U_index = np.setdiff1d(np.arange(H * W), np.concatenate([train_index, test_index]))

    print(f"[INFO] 初期 L={len(L_index)}, U={len(U_index)}, Test={len(test_index)}")

    # --- モデル定義 ---
    base_clf = SVC(kernel = "rbf", probability = True, decision_function_shape = "ovr")

    ssl_clf = SemiSupervisedClassifier(
        base_clf = base_clf,
        mode= "hybrid", # "confidence" / "spatial" / "hybrid"
        conf_threshold = 0.9,
        max_iter = 60,
        random_state = 43
    )

    # --- 学習 ---
    ssl_clf.fit(
        L_index = L_index,
        U_index = U_index,
        X = X_flat,
        y = y_flat,
        image_shape = (H, W),
        upd_LUlabel=upd_LUlabel, # spatial/hybrid モード時に必須
    )

     # --- 評価 ---
    acc_train = ssl_clf.score(X_flat[train_index], y_flat[train_index])
    acc_test = ssl_clf.score(X_flat[test_index], y_flat[test_index])

    print(f"[RESULT] Train Accuracy = {acc_train:.4f}")
    print(f"[RESULT] Test Accuracy = {acc_test:.4f}")
