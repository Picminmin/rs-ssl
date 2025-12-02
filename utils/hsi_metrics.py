import numpy as np

def compute_accuracy(y_pred, y_true, background_label = 0):
    """
    背景を除外した正解率(accuracy)を計算するシンプルな関数。
    y_pred, y_true は同じ shape の 1次元ベクトルを想定。
    """
    # 背景除外マスク
    mask = (y_true != background_label)
    # 分母
    n = np.count_nonzero(mask)
    if n == 0:
        return 0.0 # あり得ないが安全のため

    # 正解数
    correct = np.count_nonzero((y_pred == y_true) & mask)

    return correct / n

def compute_class_accuracy(y_pred, y_true, background_label = 0):
    """
    クラスごとの正答率 (CA) と平均正答率 (AA) を返す。
    Q. OAでは見落とされやすいモデルの性能を AA ではどのように補えるか。
    A. クラス不均衡、すなわち少数クラスに対するモデルの性能の良しあしを反映できる点である。
    """
    mask = (y_true != background_label)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    classes = np.unique(y_true)
    class_acc = {}

    for c in classes:
        c_mask = (y_true == c)
        n = np.count_nonzero(c_mask)
        correct = np.count_nonzero((y_pred == y_true) & c_mask)
        class_acc[c] = correct / n if n > 0 else 0.0

    ## AA
    AA = np.mean(list(class_acc.values()))
    return class_acc, AA

def compute_kappa(y_pred, y_true, background_label = 0):
    mask = (y_true != background_label)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    classes = np.unique(y_true)
    C = len(classes)

    # confusion matrix
    cm = np.zeros((C, C), dtype = int)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    for t, p in zip(y_true, y_pred):
        cm[class_to_idx[t], class_to_idx[p]] += 1

    N = np.sum(cm)
    p0 = np.trace(cm) / N # OA に相当

    # 偶然一致の確率
    pe = np.sum(np.sum(cm, axis = 0) * np.sum(cm, axis = 1)) / (N * N)

    if 1 - pe == 0:
        return 0.0

    kappa = (p0 - pe) / (1 - pe)
    return kappa

def compute_mcc(y_pred, y_true, background_label = 0):
    mask = (y_true != background_label)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    classes = np.unique(np.concatenate([y_true, y_pred]))
    C = len(classes)
    cm = np.zeros((C, C), dtype = np.int64)
    class_to_idx = {c:i for i, c in enumerate(classes)}

    for t, p in zip(y_true, y_pred):
        cm[class_to_idx[t], class_to_idx[p]] += 1

    # multi-class MCC formula
    t_sum = np.sum(cm, axis = 1)
    p_sum = np.sum(cm, axis = 0)
    N = np.sum(cm)
    c = np.trace(cm)

    # numerator
    numer = c * N - np.sum(t_sum * p_sum)

    # denominator
    denom = np.sqrt(
        (N**2 - np.sum(p_sum**2)) *
        (N**2 - np.sum(t_sum**2))
    )

    if denom == 0:
        return 0.0

    return numer / denom
