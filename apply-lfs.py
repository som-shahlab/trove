
import os
import sys
import time
import glob
import torch
import pickle
import snorkel
import argparse
import collections
from pytorch_pretrained_bert import BertTokenizer

from trove.analysis import lf_summary
from trove.data.dataloaders.ner import load_json_dataset
from trove.data.dataloaders import dataloader
from trove.labelers.core import SequenceLabelingServer
from trove.labelers.entities import *

# set reasonable pandas dataframe display defaults
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


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


def create_word_lf_mat(Xs, Ls, num_lfs):
    """
    Create word-level LF matrix from LFs indexed by sentence/word
    0 words X lfs
    1 words X lfs
    2 words X lfs
    ...

    """
    Yws = []
    for sent_i in range(len(Xs)):
        ys = dok_matrix((len(Xs[sent_i].words), num_lfs))
        for lf_i in range(num_lfs):
            for word_i, y in Ls[sent_i][lf_i].items():
                ys[word_i, lf_i] = y
        Yws.append(ys)
    return csr_matrix(vstack(Yws))


def load_labeling_functions(entity_type, sentences, top_k, data_root, active_tiers):
    """

    Parameters
    ----------
    entity_type
    sentences   unlabeled sentences used to select UMLS SABs
    top_k       top k UMLS source vocabularies (ranked by concept coverage)
    data_root

    Returns
    -------

    """
    if entity_type == 'disorder':
        labeler = DisorderLabelingFunctions(data_root)
    elif entity_type == 'drug':
        labeler = i2b2DrugLabelingFunctions(data_root)
    elif entity_type == 'shc-drug':
        labeler = DrugLabelingFunctions(data_root)
    elif entity_type == 'disease':
        labeler = DiseaseLabelingFunctions(data_root)
    elif entity_type == 'chemical':
        labeler = ChemicalLabelingFunctions(data_root)
    else:
        print(f'Fatal Error - no {entity_type}')
        sys.exit()
    return labeler.lfs(sentences, top_k, active_tiers=active_tiers)


def load_unlabeled(fpath,
                   docs,
                   n_sample_docs=50000,
                   multiplier=None,
                   use_sentences=False,
                   seed=1234):
    """ Sample unlabeled documents """
    filelist = glob.glob(f'{fpath}/*.json')
    unlabeled = dataloader(filelist)

    np.random.seed(seed)
    np.random.shuffle(unlabeled)

    # return random document sample
    if not use_sentences:
        n = multiplier * len(docs) if not n_sample_docs else n_sample_docs
        print(f'Unlabeled docs: {n}')
        return unlabeled[:n]

    # compute multiplier (e.g., we want to load 10x more sentences)
    n_train_sents = sum([len(doc.sentences) for doc in docs])

    tmp = []
    n = 0
    while n < n_train_sents * multiplier:
        doc = unlabeled.pop(0)
        n += len(doc.sentences)
        tmp.append(doc)

    print(f'Unlabeled docs: {len(tmp)}\nUnlabeled sents: {n}')


@timeit
def main(args):

    # -------------------------------------------------------------------------
    # Load Datasets
    # -------------------------------------------------------------------------
    model = 'bert-base-cased'
    tokenizer = BertTokenizer.from_pretrained(model, do_lower_case=False)
    contiguous_only = True

    # pre-canned experiment splits
    if args.indir is not None:
        data_dir = f'{args.indir}/{args.domain}/{args.corpus}/experiments/'
        prefix = f'{args.corpus}.{args.entity_type}'
        dataset = {
            name: load_json_dataset(f'{data_dir}/{name}.{prefix}.json',
                                    tokenizer,
                                    tag_fmt=args.tag_fmt,
                                    contiguous_only=contiguous_only)
            for name in ['train', 'dev', 'test']
        }
    # arbitrary train/dev/test
    else:
        dataset = {
            name: load_json_dataset(fpath,
                                    tokenizer,
                                    tag_fmt=args.tag_fmt,
                                    contiguous_only=contiguous_only)
            for name, fpath in [
                ("train", args.train),
                ("dev", args.dev),
                ("test", args.test)]
        }

    if args.unlabeled:
        # load a fixed number of unlabeled documents or some multiple of train
        dataset['unlabeled'] = load_unlabeled(args.unlabeled,
                                              dataset['train'].documents,
                                              n_sample_docs=args.n_sample_docs,
                                              multiplier=None,
                                              use_sentences=False,
                                              seed=args.seed)

    # -------------------------------------------------------------------------
    # Load Labeling Functions
    # -------------------------------------------------------------------------
    lfs = load_labeling_functions(
        args.entity_type,
        sentences=dataset['train'].sentences,
        top_k=args.top_k,
        data_root=args.data_root,
        active_tiers=args.active_tiers
    )
    print(f'Loaded {len(lfs)} LFs for entity type {args.entity_type}')

    # -------------------------------------------------------------------------
    # Apply Labeling Functions
    # -------------------------------------------------------------------------
    X_sents = [
        dataset['train'].sentences,
        dataset['dev'].sentences,
        dataset['test'].sentences
    ]
    if args.unlabeled:
        # load unlabeled documents
        X_sents.append(list(itertools.chain.from_iterable(
            [doc.sentences for doc in dataset['unlabeled']]))
        )

    for i in range(len(X_sents)):
        print(len(X_sents[i]))

    labeler = SequenceLabelingServer(num_workers=args.n_procs)
    L_sents = labeler.apply(lfs, X_sents)

    # sequence lengths
    X_seq_lens = [
        np.array([len(s.words) for s in X_sents[0]]),
        np.array([len(s.words) for s in X_sents[1]]),
        np.array([len(s.words) for s in X_sents[2]]),
        np.array([len(s.words) for s in X_sents[3]]) \
            if args.unlabeled else None
    ]

    X_doc_seq_lens = [
        np.array([len(doc.sentences) for doc in dataset['train'].documents]),
        np.array([len(doc.sentences) for doc in dataset['dev'].documents]),
        np.array([len(doc.sentences) for doc in dataset['test'].documents]),
        np.array([len(doc.sentences) for doc in dataset['unlabeled']]) \
            if args.unlabeled else None
    ]

    X_words = [
        np.array(list(
            itertools.chain.from_iterable([s.words for s in X_sents[0]]))),
        np.array(list(
            itertools.chain.from_iterable([s.words for s in X_sents[1]]))),
        np.array(list(
            itertools.chain.from_iterable([s.words for s in X_sents[2]]))),
        np.array(list(itertools.chain.from_iterable(
            [s.words for s in X_sents[3]]))) if args.unlabeled else None
    ]

    Y_words = [
        [dataset['train'].tagged(i)[-1] for i in range(len(dataset['train']))],
        [dataset['dev'].tagged(i)[-1] for i in range(len(dataset['dev']))],
        [dataset['test'].tagged(i)[-1] for i in range(len(dataset['test']))],
        [],
    ]

    tag2idx = {'O': 2, 'I': 1, 'B': 3}

    Y_words[0] = np.array(
        [tag2idx[t[0]] for t in list(itertools.chain.from_iterable(Y_words[0]))])
    Y_words[1] = np.array(
        [tag2idx[t[0]] for t in list(itertools.chain.from_iterable(Y_words[1]))])
    Y_words[2] = np.array(
        [tag2idx[t[0]] for t in list(itertools.chain.from_iterable(Y_words[2]))])
    Y_words[3] = np.array([])

    L_words = [
        create_word_lf_mat(X_sents[0], L_sents[0], len(lfs)),
        create_word_lf_mat(X_sents[1], L_sents[1], len(lfs)),
        create_word_lf_mat(X_sents[2], L_sents[2], len(lfs)),
        create_word_lf_mat(X_sents[3], L_sents[3], len(lfs)) \
            if len(L_sents) > 3 else None
    ]

    # print summary of LF performance on dev data
    lf_names = [lf.name for lf in lfs]

    # BIO hack
    for i in range(len(Y_words)):
        freq = collections.Counter()
        for y in Y_words[i]:
            freq[y] += 1
        print(i, freq)

    if len(np.unique(Y_words[0])) != 2:
        print("Converting BIO to IO for LF metrics")
        Y_train = np.array([2 if y == 2 else 1 for y in Y_words[0]])
        Y_dev = np.array([2 if y == 2 else 1 for y in Y_words[1]])
        Y_test = np.array([2 if y == 2 else 1 for y in Y_words[2]])

        tmp = [Y_train, Y_dev, Y_test]
        for i in range(len(tmp)):
            freq = collections.Counter()
            for y in tmp[i]:
                freq[y] += 1
            print(i, freq)

    else:
        Y_train = Y_words[0]
        Y_dev = Y_words[1]
        Y_test = Y_words[2]

    print('TRAIN\n' + ('=' * 80))
    print(lf_summary(L_words[0], Y=Y_train, lf_names=lf_names))
    print('DEV\n' + ('=' * 80))
    print(lf_summary(L_words[1], Y=Y_dev, lf_names=lf_names))
    print('TEST\n' + ('=' * 80))
    print(lf_summary(L_words[2], Y=Y_test, lf_names=lf_names))

    # -------------------------------------------------------------------------
    # Dump Matrices
    # -------------------------------------------------------------------------
    pickle.dump(X_sents, open(f'{args.outdir}/X_sents','wb'))
    pickle.dump(X_seq_lens, open(f'{args.outdir}/X_seq_lens','wb'))
    pickle.dump(X_doc_seq_lens, open(f'{args.outdir}/X_doc_seq_lens','wb'))

    pickle.dump(X_words, open(f'{args.outdir}/X_words', 'wb'))
    pickle.dump(Y_words, open(f'{args.outdir}/Y_words', 'wb'))
    pickle.dump(L_words, open(f'{args.outdir}/L_words', 'wb'))
    pickle.dump(lf_names, open(f'{args.outdir}/LF_names', 'wb'))


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--indir", type=str, default=None)
    parser.add_argument("--train", type=str, default=None)
    parser.add_argument("--dev", type=str, default=None)
    parser.add_argument("--test", type=str, default=None)

    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--unlabeled", type=str, default=None)
    parser.add_argument("--n_sample_docs", type=int, default=25000)
    parser.add_argument("--active_tiers", type=int, default=1234)

    parser.add_argument("--entity_type", type=str, default=None)
    parser.add_argument("--domain", type=str, default=None)
    parser.add_argument("--corpus", type=str, default=None)
    parser.add_argument("--tag_fmt", type=str, default='IO')
    parser.add_argument("--top_k", type=int, default=10)

    parser.add_argument("--n_procs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir, exist_ok=True)
        print(f'Created output directory {args.outdir}')

    args.active_tiers = list(map(int, list(str(args.active_tiers))))

    print(sys.version_info)
    print(f'PyTorch v{torch.__version__}')
    print(f'Snorkel v{snorkel.__version__}')
    main(args)



# python apply-lfs.py \
# --indir /Users/fries/Desktop/NER-Datasets-ALL/ \
# --outdir /Users/fries/Desktop/label_matrices_inkfish_final/disorder/rand/ \
# --data_root /Users/fries/Code/refactor/inkNER/data/supervision/ \
# --unlabeled /Users/fries/Desktop/NER-Datasets-ALL/ehr/mimic3/experiments/ \
# --corpus shareclef \
# --n_sample_docs 25000 \
# --active_tiers 12 \
# --domain ehr \
# --tag_fmt IO \
# --entity_type disorder \
# --top_k 5

#
# python apply-lfs.py \
# --indir /Users/fries/Desktop/NER-Datasets-ALL/ \
# --outdir /Users/fries/Desktop/label_matrices_inkfish/disorder/rand/ \
# --data_root /Users/fries/Code/refactor/inkNER/data/supervision/ \
# --corpus shareclef \
# --domain ehr \
# --tag_fmt IO \
# --entity_type disorder \
# --top_k 5

# python apply-lfs.py \
# --indir /Users/fries/Desktop/NER-Datasets-ALL/ \
# --outdir /Users/fries/Desktop/label_matrices_inkfish/chemical/BC5CDR/ \
# --data_root /Users/fries/Code/refactor/inkNER/data/supervision/ \
# --corpus cdr \
# --domain pubmed \
# --tag_fmt IO \
# --entity_type chemical \
# --top_k 10

# COVID Experimemts

# python apply-lfs.py \
# --train /Users/fries/Desktop/covid-train/final/train.shc-covid.json \
# --dev /Users/fries/Desktop/NER-Datasets-ALL/ehr/i2b2meds/experiments/dev.i2b2meds.drug.json \
# --test /Users/fries/Desktop/covid-gold-docs/gold/drug/covid-shc.drug.gold.json \
# --data_root /Users/fries/Code/refactor/inkNER/data/supervision/ \
# --outdir /Users/fries/Desktop/covid-gold-docs/label_matrices/drug/ \
# --active_tiers 123 \
# --tag_fmt IO \
# --entity_type shc-drug \
# --top_k 2


# python apply-lfs.py \
# --train /Users/fries/Desktop/covid-train/final/train.shc-covid.json \
# --dev /Users/fries/Desktop/NER-Datasets-ALL/ehr/shareclef/experiments/dev.shareclef.disorder.json \
# --test /Users/fries/Desktop/covid-gold-docs/gold/diso_sign_symp/covid-shc.diso_sign_symp.gold.json \
# --data_root /Users/fries/Code/refactor/inkNER/data/supervision/ \
# --outdir /Users/fries/Desktop/covid-gold-docs/label_matrices/diso_sign_symp/ \
# --active_tiers 123 \
# --tag_fmt IO \
# --entity_type disorder \
# --top_k 6



