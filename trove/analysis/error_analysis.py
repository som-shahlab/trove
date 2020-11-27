import numpy as np
from .metrics import score_sequences, tokens_to_sequences


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


def get_coverage(L, Y=None):
    mask = np.all(L == -1, axis=1)
    if np.any(Y):
        Y_hat = list(Y[mask])
        class_coverage = []
        for y in sorted(np.unique(Y)):
            c = [Y_hat.count(y), list(Y).count(y)]
            class_coverage.append(c)
        for i, c in enumerate(class_coverage):
            print(f'y={i} abstained {c[0] / c[1] * 100:2.1f}% ({c[0]}/{c[1]})')

    coverage = 1.0 - (np.where(mask == 1)[0].shape[0] / mask.shape[0])
    print(f'Coverage: {coverage * 100:2.1f}%')


def eval_label_model(model, L, Y, seq_lens):

    idx2tag = {0: 'O', 1: 'I-X', 2: 'B-X'}

    # label model
    y_pred = model.predict(L)
    scores = score_sequences(*tokens_to_sequences(Y, y_pred, seq_lens, idx2tag=idx2tag))
    print('[Label Model]   {}'.format(
        ' | '.join([f'{m}: {v * 100:2.2f}' for m, v in scores.items()]))
    )

    # MV baseline
    y_pred = mv(L, 0)
    scores = score_sequences(*tokens_to_sequences(Y, y_pred, seq_lens, idx2tag=idx2tag))
    print('[Majority Vote] {}'.format(
        ' | '.join([f'{m}: {v * 100:2.2f}' for m, v in scores.items()]))
    )