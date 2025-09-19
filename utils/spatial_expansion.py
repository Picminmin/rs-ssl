
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
    空間的隣接性に基づいて L, U を更新する。

    Args:
        pre_L (np.ndarray): ラベル付きデータのインデックス配列
        pre_U (np.ndarray): ラベルなしデータのインデックス配列
        y_pred (np.ndarray): ラベルなしデータに対する予測ラベル
        expand_label (np.ndarray): 拡張されたラベルマップ (1次元 flatten)
        ground_truth (np.ndarray): GTラベル (1次元 flatten)
        background_label (int, optional): 背景ラベル
        connectivity (int, optional): 4近傍 or 8近傍

    Returns:
        upd_L (np.ndarray): 更新後のラベル付きデータのインデックス
        upd_U (np.ndarray): 更新後のラベルなしデータのインデックス
        upd_flag (bool): 更新が発生したか
        expand_label (np.ndarray): 更新後の拡張ラベルマップ
    """
    height, width = ground_truth.shape
    remove_index = []
    upd_flag = False

    # L の周囲にある U の候補を探索
    candidate_neighbors = chain.from_iterable(
        get_neighbors(idx, width, height, connectivity) for idx in pre_L
    )
    candidate_neighbors = np.unique([i for i in candidate_neighbors if i in pre_U])

    for u_idx in candidate_neighbors:
        # u_idx の近傍にあるラベル分布を集計
        neighbors = get_neighbors(u_idx, width, height, connectivity)
        class_votes = np.zeros(ground_truth.max() + 1, dtype=int)

        for n_idx in neighbors:
            if expand_label[n_idx] != background_label:
                cls = expand_label[n_idx]
                class_votes[cls] += 1

        # 投票があれば最頻クラスを割り当てる
        if class_votes.sum() > 0:
            best_class = np.argmax(class_votes[1:]) + 1 # 背景(0)を除外
            # 同数票の処理
            if np.sum(class_votes == class_votes[best_class]) == 1:
                expand_label[u_idx] = best_class
                remove_index.append(u_idx)
                upd_flag = True

        # L と U を更新
        if remove_index:
            upd_L = np.concatenate([pre_L, np.array(remove_index)])
            upd_U = np.array([i for i in pre_U if i not in remove_index])
        else:
            upd_L, upd_U = pre_L, pre_U

        return upd_L, upd_U, upd_flag, expand_label
