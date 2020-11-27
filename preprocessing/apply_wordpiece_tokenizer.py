#
# USAGE
#
# python apply_wordpiece_tokenizer.py \
# --inputdir /data5/stride8_nlp_notes/sentence_corpus/ \
# --outputdir /data5/stride8_nlp_notes/sentence_corpus_wp/ \
# --model /data5/stride8_nlp_notes/wordpiece_model/bert-wordpiece-vocab.txt \
# --n_procs 64
#
# Train wordpiece tokenizer time:
# 19:56:01 hours
#
# Time apply tokenizer all Stanford notes
# 64 cores
# 20891.7037 sec (5.8 hours)
#
#

# ./fasttext skipgram \
# -input /data5/stride8_nlp_notes/shc.sentences.wp.2020_04_11.txt \
# -output /data5/stride8_nlp_notes/shc.wp.d128.w10.neg20.epochs01.fasttext \
# -minCount 0 \
# -epoch 1 \
# -neg 20 \
# -thread 96 \
# -dim 128 \
# -lr 0.05 \
# -ws 10 \
# -wordNgrams 1

import glob
import sys
import time
import argparse
import numpy as np
from joblib import Parallel, delayed
from functools import partial
from transformers import BertTokenizer
from toolz import partition

def timeit(f):
    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print(f'{f.__name__} took: {te - ts:2.4f} sec')
        return result
    return timed


def corpus(fpath, encoding):
    with open(fpath, 'r', encoding=encoding) as fp:
        for line in fp:
            yield line.strip('\n')


def transform_texts(tokenizer, batch, output_dir, encoding='utf-8'):
    batch = [fpath for fpath in batch if fpath is not None]
    for fpath in batch:
        outpath = f"{output_dir}/{fpath.split('/')[-1]}.wp"
        print("Processing batch", fpath)

        n = 0
        with open(outpath, "w", encoding=encoding) as fp:
            for line in corpus(fpath, encoding=encoding):
                try:
                    toks = tokenizer.tokenize(line)
                    fp.write(f"{' '.join(toks)}\n")
                    n += 1
                except Exception as e:
                    print(f'Error {e} -- skipping line')

        print(f"Wrote {n} lines to {outpath}")

@timeit
def main(args):
    # load files
    filelist = glob.glob(f'{args.inputdir}/*')
    block_size = int(np.ceil(len(filelist) / 64))
    partitions = partition(block_size, filelist, pad=None)
    print(f'Loading {len(filelist)} files')

    # load word piece model
    tokenizer = BertTokenizer.from_pretrained(
        args.model, do_lower_case=args.do_lower_case
    )

    executor = Parallel(n_jobs=args.n_procs,
                        backend="multiprocessing",
                        prefer="processes")
    do = delayed(partial(transform_texts, tokenizer))

    tasks = (do(batch, args.outputdir, args.encoding) for batch in partitions)
    executor(tasks)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-i", "--inputdir", type=str, default=None,
                           help="input directory")
    argparser.add_argument("-o", "--outputdir", type=str, default=None,
                           help="output directory")

    argparser.add_argument("-m", "--model", type=str, default=None,
                           help="word piece token model")
    argparser.add_argument("-e", "--encoding", type=str, default="utf-8",
                           help="encoding")
    argparser.add_argument("-l", "--do_lower_case", action="store_true")
    argparser.add_argument("-n", "--n_procs", type=int, default=2,
                           help="number of processes")

    args = argparser.parse_args()

    print(f'Python:      {sys.version}')
    for attrib in args.__dict__.keys():
        v = 'None' if not args.__dict__[attrib] else args.__dict__[attrib]
        print("{:<15}: {:<10}".format(attrib, v))

    main(args)
