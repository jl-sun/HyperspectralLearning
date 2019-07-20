import numpy as np


def confusion_matrix(pred, truth):
    tp = fn = fp = tn = 0
    for i, j in zip(pred, truth):
        if i == 0 and j == 0:
            tn += 1
        elif i == 0 and j == 1:
            fn += 1
        elif i == 1 and j == 0:
            fp += 1
        elif i == 1 and j == 1:
            tp += 1
        else:
            raise ValueError("prediction and truth value should be 0 or 1.")
    return dict(TP=tp, FN=fn, FP=fp, TN=tn)


def roc_curve(score, truth):
    if not isinstance(score, np.ndarray) \
        or not isinstance(truth, np.ndarray):
        raise TypeError("You should send two np.ndarray as arguments.")
    if len(score) != len(truth):
        raise ValueError("Length if two arguments should be same.")
    thresholds = np.sort(score)[::-1]
    xs = []
    ys = []
    for th in thresholds:
        pred = 1 * (score > th)
        m = confusion_matrix(pred, truth)
        fpr = m['FP'] / (m['FP'] + m['TN'])
        tpr = m['TP'] / (m['TP'] + m['FN'])
        xs.append(fpr)
        ys.append(tpr)
    xs.append(1.0)
    ys.append(1.0)
    return xs, ys


if __name__ == '__main__':
    d = np.array([0.9, 0.8, 0.7, 0.6, 0.55, 0.54, 0.53, 0.52, 0.51, 0.505, 0.4, 0.39, 0.38, 0.37, 0.36, 0.35, 0.34, 0.33, 0.30, 0.1])
    t = np.array([1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0])
    x, y = roc_curve(d, t)
    print(x, y)
