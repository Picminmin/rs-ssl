
import numpy as np
from itertools import chain

def get_neighbors(index: int, width: int, height: int, connectivity: int = 8) -> list[int]:
    """
    指定された画素インデックスの隣接画素(4近傍 or 8近傍)を返す。

    Args:
        index (int): 画像をflatten した時のインデックス
        width (int): 画像の幅
        height (int): 画像の高さ
        connectivity (int): 近傍の種類 (4 または 8)
    """
    y, x = divmod(index, width) # index → (row, col)

    if connectivity == 4:
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),           (0, 1),
                      (1, -1),  (1, 0) , (1, 1)]

    neighbors = []
    for dy, dx in directions:
        ny, nx = y + dy, x + dx
        if 0 <= ny < height and 0 <= nx < width:
            neighbors.append(ny * width + nx)

    return neighbors

def upd_LUlabel(
    pre_L: np.ndarray,
    pre_U: np.ndarray,
    y_pred: np.ndarray,
    expand_label: np.ndarray,
    ground_truth: np.ndarray,
    background_label: int = 0,
    connectivity: int = 8
):
    """
    空間的隣接性に基づいて L, U を更新する。ただし boundary_mask 上の画素は拡張しない。

    Args:
        pre_L (np.ndarray): ラベル付きデータのインデックス配列
        pre_U (np.ndarray): ラベルなしデータのインデックス配列
        y_pred (np.ndarray): ラベルなしデータに対する予測ラベル
        expand_label (np.ndarray): 拡張されたラベルマップ 1D: 長さ H*W, 0は背景/未ラベル
        ground_truth (np.ndarray): GTラベル 2D: 形状(H, W)を想定
        background_label (int, optional): 背景ラベル
        connectivity (int, optional): 4近傍 or 8近傍

    Returns:
        upd_L (np.ndarray): 更新後のラベル付きデータのインデックス
        upd_U (np.ndarray): 更新後のラベルなしデータのインデックス
        upd_flag (bool): 更新が発生したか
        expand_label (np.ndarray): 更新後の拡張ラベルマップ
    """

    if ground_truth.ndim != 2:
        raise ValueError("ground_truthは2D (H, W)を想定。")

    H, W = ground_truth.shape
    remove_index = []
    upd_flag = False

    # U を集合にしてmembershipを高速化
    U_set = set(pre_U.tolist())

    # L の8近傍にある U を候補化
    cand = set()
    for idx in pre_L:
        for nb in get_neighbors(idx, W, H, connectivity):
            if nb in U_set:
                cand.add(nb)

    # 評価カウント用のクラス数(背景を含む)
    n_classes = int(max(expand_label.max(), ground_truth.max())) + 1

    for u_idx in cand:

        nbs = get_neighbors(u_idx, W, H, connectivity)
        # 背景以外の既ラベルだけを見る
        labels = [expand_label[n] for n in nbs if expand_label[n] != background_label]
        if not labels:
            continue

        votes = np.bincount(labels, minlength = n_classes)
        votes[background_label] = 0 # 背景は無視

        best = int(votes.argmax())
        top = votes[best]
        # 2番目との比較で「同数票かどうか」を判定
        votes_wo_best = votes.copy()
        votes_wo_best[best] = -1
        second = votes_wo_best.max()

        if top > 0 and top > second: # ← 同数票なし & 票が入っている
            expand_label[u_idx] = best
            remove_index.append(u_idx)
            upd_flag = True

    # --- ここで一括更新 (← 重要: for の外) ---
    if remove_index:
        remove_index = np.array(remove_index, dtype = int)
        upd_L = np.concatenate([pre_L, remove_index])
        # pre_U から remove_index を引く
        upd_U = np.setdiff1d(pre_U, remove_index, assume_unique = False)
    else:
        upd_L, upd_U = pre_L, pre_U

    return upd_L, upd_U, upd_flag, expand_label
