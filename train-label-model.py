import os
import sys
import time
import torch
import pickle
import snorkel
import trove
import argparse
import functools
import itertools
import collections
import numpy as np
#from snorkel.labeling import LabelModel
from snorkel.labeling.model.label_model import LabelModel

from trove.analysis.metrics import split_by_seq_len
from trove.analysis.error_analysis import get_coverage, eval_label_model, mv, smv
from trove.models.model_search import grid_search
from sklearn.metrics import *
import pickle
import matplotlib.pyplot as plt

def timeit(f):
    """
    Decorator for timing function calls
    :param f:
    :return:
    """
    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print(f'{f.__name__} took: {te - ts:2.4f} sec')
        return result

    return timed

def convert_label_matrix(L):
    # abstain is -1
    # negative is 0
    L = L.toarray().copy()
    L[L == 0] = -1
    L[L == 2] = 0
    return L


def sample_doc_sents(xs, ys, x_seq_lens, x_doc_seq_lens, n_docs, seed=1234):
    """
    Sample sentences by document. This is to measure how many
    labeled documents are need for label model tuning.
    """
    # bin sentences
    X = split_by_seq_len(xs, x_seq_lens)
    Y = split_by_seq_len(ys, x_seq_lens) if np.any(ys) else None

    # bin sentences by document
    docs = split_by_seq_len(X, x_doc_seq_lens)
    tags = split_by_seq_len(Y, x_doc_seq_lens) if Y is not None else None

    # sample sentences binned by document
    np.random.seed(seed)
    idxs = np.random.choice(range(0, len(docs)), n_docs, replace=False)
    X_hat = list(itertools.chain.from_iterable([docs[i] for i in idxs]))
    print(f"Sampled doc indices: {idxs}")

    if tags is not None:
        Y_hat = list(itertools.chain.from_iterable([tags[i] for i in idxs]))
        Y_hat = np.concatenate(Y_hat)
    else:
        Y_hat = None

    # determine sequence lengths
    X_hat_seq_lens = np.array([len(doc) for doc in X_hat])
    X_hat = np.vstack(X_hat)
    print(f"Sampled {len(X_hat_seq_lens)} sentences")
    return (X_hat, Y_hat, X_hat_seq_lens)


def get_token_scores(y_gold, y_pred, name=''):
    p = precision_score(y_gold, y_pred) * 100
    r = recall_score(y_gold, y_pred) * 100
    f1 = f1_score(y_gold, y_pred) * 100
    print(f'{name} | {p:2.2f} {r:2.2f} {f1:2.2f}')
    return (p,r,f1)


def plot_prc(fpath, scores):
    plt.step(scores['LM']['recall'], scores['LM']['precision'], where='post')
    plt.step(scores['SMV']['recall'], scores['SMV']['precision'], where='post')

    ap_smv = scores['SMV']['avg_precision_score']
    ap_lm = scores['LM']['avg_precision_score']

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(
        f'Average Precision Score SMV:{ap_smv * 100:2.2f} LM:{ap_lm * 100:2.2f}')
    plt.savefig(fpath, dpi=300, format='pdf')
    plt.close()


def dump_avg_precision_scores(root_fpath, gold, probas):

    scores = {}
    for name in probas:
        precision, recall, thresholds = precision_recall_curve(gold, probas[name])
        scores[name] = {'precision':precision, 'recall':recall, 'thresholds':thresholds}
        scores[name]['avg_precision_score'] = average_precision_score(gold, probas[name], average=None, pos_label=1)
        scores[name]['auroc'] = roc_auc_score(gold, probas[name], average=None)
        print(f'{name} | avg_precision_score', scores[name]['avg_precision_score'])
        print(f'{name} | auroc', scores[name]['auroc'])

    # dump scores
    pickle.dump(scores, open(f'{root_fpath}.scores.pkl','wb'))

    # dump plot
    plot_prc(f'{root_fpath}.avg_precision_curves.pdf', scores)
    print(f'{root_fpath} DONE')


@timeit
def main(args):

    # -------------------------------------------------------------------------
    # Load Dataset and L Matrices
    # -------------------------------------------------------------------------
    X_sents = pickle.load(open(f'{args.indir}/X_sents', 'rb'))
    X_seq_lens = pickle.load(open(f'{args.indir}/X_seq_lens', 'rb'))
    X_doc_seq_lens = pickle.load(open(f'{args.indir}/X_doc_seq_lens', 'rb'))
    L_words = pickle.load(open(f'{args.indir}/L_words', 'rb'))
    X_words = pickle.load(open(f'{args.indir}/X_words', 'rb'))
    Y_words = pickle.load(open(f'{args.indir}/Y_words', 'rb'))

    # display tag statistics
    for i in range(len(Y_words)):
        freq = collections.Counter()
        for y in Y_words[i]:
            freq[y] += 1
        print(i, freq)

    for i in range(len(L_words)):
        size = L_words[i].shape if L_words[i] is not None else None
        print(f'i={i} {size}')
    print(f'train={args.train} dev={args.dev} test={args.test}')

    # transform label matrices to Snorkel v9.3+ format
    if args.label_fmt == 'snorkel':
        L_words = [
            convert_label_matrix(L_words[0]),
            convert_label_matrix(L_words[1]),
            convert_label_matrix(L_words[2]),
            convert_label_matrix(L_words[3]) \
                if L_words[3] is not None else None
        ]

        remap_tags = {2:0, 1:1, 3:2}
        Y_words = [
            np.array([remap_tags[y] for y in Y_words[0]]),
            np.array([remap_tags[y] for y in Y_words[1]]),
            np.array([remap_tags[y] for y in Y_words[2]]),
            np.array([])
        ]
        print("Coverted to Snorkel 9.3+ label matrix format")

    # subsample training set
    if args.n_train_docs:
        L_train = sample_doc_sents(L_words[args.train],
                                   Y_words[args.train],
                                   X_seq_lens[args.train],
                                   X_doc_seq_lens[args.train],
                                   n_docs=args.n_train_docs,
                                   seed=args.data_seed)
        L_train, Y_train, train_seq_lens = L_train
        print(f'Sampled {args.n_train_docs} TRAIN documents')
    else:
        L_train = L_words[args.train]
        Y_train = Y_words[args.train]
        train_seq_lens = X_seq_lens[args.train]

    if args.n_dev_docs:
        L_dev = sample_doc_sents(L_words[args.dev],
                                 Y_words[args.dev],
                                 X_seq_lens[args.dev],
                                 X_doc_seq_lens[args.dev],
                                 n_docs=args.n_dev_docs,
                                 seed=args.data_seed)
        L_dev, Y_dev, dev_seq_lens = L_dev
        print(f'Sampled {args.n_dev_docs} DEV documents')
    else:
        L_dev = L_words[args.dev]
        Y_dev = Y_words[args.dev]
        dev_seq_lens = X_seq_lens[args.dev]

    print(f'TRAIN L : {L_train.shape}')
    print(f'DEV L   : {L_dev.shape}')

    # -------------------------------------------------------------------------
    # Train Label Model
    # -------------------------------------------------------------------------
    # TODO Add FlyingSquid label model back in

    np.random.seed(args.seed)
    n = L_train.shape[0]

    param_grid = {
        'lr': [0.01, 0.005, 0.001, 0.0001],
        'l2': [0.001, 0.0001],
        'n_epochs': [50, 100, 200, 600, 700, 1000],
        'prec_init': [0.6, 0.7, 0.8, 0.9],
        'optimizer': ["adamax"],
        'lr_scheduler': ['constant'],
        'mu_eps': [1 / 10 ** np.ceil(np.log10(n * 100)),
                   1 / 10 ** np.ceil(np.log10(n * 10)),
                   1 / 10 ** np.ceil(np.log10(n))],
        'seed': list(np.random.randint(0, 10000, 400))
    }

    model_class_init = {
        'cardinality': 2,
        'verbose': True
    }

    num_hyperparams = functools.reduce(
        lambda x,y:x*y, [len(x) for x in param_grid.values()]
    )
    print("Hyperparamater Search Space:", num_hyperparams)

    model, best = grid_search(LabelModel,
                              model_class_init,
                              param_grid,
                              train=(L_train, Y_train, train_seq_lens),
                              dev=(L_dev, Y_dev, dev_seq_lens),
                              n_model_search=args.n_model_search,
                              val_metric=args.val_metric,
                              seq_eval=not args.token_eval,
                              tag_fmt_ckpnt=args.tag_fmt_ckpnt,
                              checkpoint_gt_mv=args.checkpoint_gt_mv)

    # evaluate best model performance on all labeled splits
    metrics = ['precision', 'recall', 'f1', 'accuracy']
    for i in range(3):
        if not np.any(Y_words[i]):
            continue
        print(f'Split: {i}')
        y_hat = np.array([0 if y == 0 else 1 for y in Y_words[i]])
        get_coverage(L_words[i], y_hat)
        print("IO")
        eval_label_model(model, L_words[i], y_hat, X_seq_lens[i])
        print('-' * 80)
        print("BIO")
        eval_label_model(model, L_words[i], Y_words[i], X_seq_lens[i])
        print('-' * 80)

    # save final trained model
    if args.outdir:
        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)
        ts = int(time.time())
        checkpoint_fpath = f'model.n{L_train.shape[0]}.{ts}.pkl'
        model.save(f'{args.outdir}/{checkpoint_fpath}')
        print(f'Saved label model to: {args.outdir}/{checkpoint_fpath}')

        # save results
        for i in range(3):

            if not np.any(Y_words[i]):
                continue

            #y_gold = Y_words[i]
            y_gold = np.array([0 if y == 0 else 1 for y in Y_words[i]])

            y_pred = model.predict(L_words[i])
            y_pred_mv = mv(L_words[i], break_ties=0)

            y_proba = model.predict_proba(L_words[i])[..., 1]
            y_proba_smv = smv(L_words[i])[..., 1]

            # evaluate token-level metrics
            lm_scores = get_token_scores(y_gold, y_pred, name='LM')
            mv_scores = get_token_scores(y_gold, y_pred_mv, name='MV')

            outfpath = f'{args.outdir}/split.{i}.avg_prec_score'
            dump_avg_precision_scores(outfpath, y_gold,
                                      {'LM': y_proba, 'SMV': y_proba_smv})


if __name__=="__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--indir", type=str, default=None)
    parser.add_argument("--outdir", type=str, default=None)

    parser.add_argument("--train", type=int, default=None)
    parser.add_argument("--dev", type=int, default=None)
    parser.add_argument("--test", type=int, default=None)
    parser.add_argument("--label_fmt", type=str, default='snorkel')

    parser.add_argument("--token_eval", action="store_true")
    parser.add_argument("--tag_fmt_ckpnt", type=str, default='IO')

    parser.add_argument("--n_train_docs", type=int, default=None)
    parser.add_argument("--n_dev_docs", type=int, default=None)
    parser.add_argument("--n_model_search", type=int, default=50)
    parser.add_argument("--val_metric", type=str, default='f1')
    parser.add_argument("--checkpoint_gt_mv", action="store_true")


    parser.add_argument("--data_seed", type=int, default=1234)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    print(sys.version_info)
    print(f'PyTorch v{torch.__version__}')
    print(f'Snorkel v{snorkel.__version__}')
    print(f'Inkfish v{trove.__version__}')
    print(args)
    main(args)


# python train-label-model.py \
# --indir /Users/fries/Desktop/label_matrices_inkfish/chemical/BC5CDR/ \
# --train 0 \
# --dev 1 \
# --test 2 \
# --n_model_search 20 \
# --data_seed 1234 \
# --seed 1234


# python train-label-model.py \
# --indir /Users/fries/Desktop/label_matrices_inkfish/disorder/CLEF/ \
# --train 0 \
# --dev 1 \
# --test 2 \
# --n_model_search 500 \
# --data_seed 1234 \
# --seed 1234

# COVID

# python train-label-model.py \
# --indir /Users/fries/Desktop/covid-gold-docs/label_matrices/drug/1234/ \
# --outdir /Users/fries/Desktop/covid-gold-docs/label_models/drug/1234 \
# --train 0 \
# --dev 1 \
# --test 2 \
# --n_model_search 100 \
# --data_seed 1234 \
# --seed 1234


# python train-label-model.py \
# --indir /Users/fries/Desktop/covid-gold-docs/label_matrices/diso_sign_symp/123/ \
# --outdir /Users/fries/Desktop/covid-gold-docs/label_models/diso_sign_symp/123/ \
# --train 0 \
# --dev 1 \
# --test 2 \
# --n_model_search 100 \
# --data_seed 1234 \
# --seed 1234
