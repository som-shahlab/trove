import numpy as np


def mv(L, break_ties, abstain=-1):
    """Simple majority vote"""
    from statistics import mode
    y_hat = []
    for row in L:
        # get non abstain votes
        row = row[row != abstain]
        try:
            l = mode(row)
        except:
            l = break_ties
        y_hat.append(l)
    return np.array(y_hat).astype(np.int)

def smv(L, abstain=-1, uncovered=0):
    """Soft majority vote"""
    y_hat = []
    k = np.unique(L[L != abstain]).astype(int)
    k = list(range(min(k), max(k) + 1))
    for row in L:
        # get non abstain votes
        row = list(row[row != abstain])
        N = len(row)
        if not N:
            y_hat.append([1.0, 0])
        else:
            p = []
            for i in k:
                p.append(row.count(i) / N)
            y_hat.append(p)
    return np.array(y_hat).astype(np.float)