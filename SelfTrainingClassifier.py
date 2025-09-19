import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from RS_GroundTruth.rs_dataset import RemoteSensingDataset  # あなたのrs_dataset.py
from spatialcv.spatial_train_test_split import spatial_train_test_split, SpatialSplitConfig

class SelfTrainingClassifier(BaseEstimator, ClassifierMixin):
    """
    簡単な自己訓練(Self-Training)の雛形クラス
    scikit-learn API (BaseEstimator, ClassifierMixin) 準拠
    """
    def __init__(self, base_clf, max_iter=10, conf_threshold=0.9, random_state = None):
        """
        Args:
            base_clf : scikit-learnの分類器 (SVC, RandomForest など)
            max_iter (int): 自己訓練の最大反復回数
            conf_threshold (float): 疑似ラベルを追加する信頼度の閾値
            random_state (int): 乱数シード
        """
        self.base_clf = base_clf
        self.max_iter = max_iter
        self.conf_threshold = conf_threshold
        self.random_state = random_state
        self.history = []

    def fit(self, L, U):
        """
        Labeled データ(L)とUnlabeledデータ(U)を使って学習
        Args:
            L (list of tuple): [(x1, y1), (x2, y2), ...] 形式のラベル付きデータ
            U (ndarray): shape = (n_unlabeled, n_features) のラベルなしデータ
        """

        # rng = np.random.RandomState(self.random_state)
        L = list(L) # コピー
        U = np.array(U)

        for it in range(self.max_iter):
            print(f"[INFO] Iteration {it+1}/{self.max_iter}, L size={len(L)}, U size={len(U)}")

            if len(L) == 0 or len(U) == 0:
                break

            # --- Lを学習に使う ---
            X_L, y_L = map(np.array,(zip(*L)))
            self.base_clf.fit(X_L, y_L)

            # --- Uに対して予測 ---
            probs = self.base_clf.predict_proba(U)
            y_pred = probs.argmax(axis = 1)
            conf = probs.max(axis = 1)

            # --- 高信頼度サンプルを追加 ---
            mask = conf >= self.conf_threshold
            if not np.any(mask):
                print("[INFO] No confident pseudo-labels → Stop")
                break

            new_X = U[mask]
            new_y = y_pred[mask]

            # Lに追加
            L.extend(list(zip(new_X, new_y)))

            # Uから削除
            U = U[~mask]

            self.history.append((len(L), len(U)))
            # --- 最終モデル ---
            X_L, y_L = map(np.array, zip(*L))
            self.base_clf.fit(X_L, y_L)
            return self

    def predict(self, X):
        return self.base_clf.predict(X)
    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


if __name__ == "__main__":
    from sklearn.svm import SVC
    from dataclasses import dataclass

    # データ読み込み
    ds = RemoteSensingDataset(remove_bad_bands=True)
    X, y = ds.load("Indianpines")

    random_state = 43
    @dataclass
    class SpatialSplitConfig:
        n_rows: int = 13                       # ブロック分割数(縦)
        n_cols: int = 13                       # ブロック分割数(横)
        min_train_samples_per_class: int = 5  # 各土地被覆クラスで作成する教師データのサンプル数の下限
        min_test_samples_per_class: int = 5    # 各土地被覆クラスで作成するなテストサンプル数の下限
        background_label: int = 0              # 各土地被覆クラスで作成するなテストサンプル数の下限
        random_state: int = random_state       # 乱数シード
        auto_adjust_test_size: bool = True    # テストサイズを自動調整するか
        min_search_test_ratio: float = None    # 自動探索モード時の下限
        max_search_test_ratio: float = None    # 自動探索モード時の上限
        step: float = 0.05                     # 自動探索モード時の刻み幅
        max_iter: int = 100                    # ランダム試行回数

    # テストサイズの指定
    test_size ,error_rate = 0.6, 0.1
    cfg = SpatialSplitConfig(min_search_test_ratio = test_size - error_rate,
                             max_search_test_ratio = test_size + error_rate)
    X_train, X_test, y_train, y_test, train_mask, test_mask, best_ts = spatial_train_test_split(
        X = X, y = y, test_size = test_size, cfg = cfg
    )

    # --- L, Uの作成 ---
    # Labeled = train_mask 内の教師データ
    L = list(zip(X_train, y_train))

    # Unlabeled = 背景(0)も含めた「train_mask ∨ test_mask」
    all_mask = np.logical_or(train_mask, test_mask)
    U = X.reshape(-1, X.shape[2])[all_mask.flatten()]
    U = np.array([u for u in U if u.tolist() not in [l[0].tolist() for l in L]])

    # --- モデル定義 ---
    base_clf = SVC(kernel="rbf", probability=True)  # probaが必要ならprobability=True

    # 予測結果の信頼度の閾値の設定
    max_iter, conf_threshold = 10, 0.01
    ssl_clf = SelfTrainingClassifier(base_clf=base_clf,
                                     max_iter=max_iter,
                                     random_state = random_state,
                                     conf_threshold=conf_threshold
    )

    ssl_clf.fit(L, U)

    # --- 評価 ---
    X_test_flat = X_test
    y_test_flat = y_test
    acc = ssl_clf.score(X_test_flat, y_test_flat)
    print(f"[RESULT] Test Accuracy = {acc:.4f}")
