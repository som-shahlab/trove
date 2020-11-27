import os
import sys
import glob
import json
import time
import logging
import argparse
import pandas as pd
from pathlib import Path
from joblib import Parallel, delayed
from functools import partial
from spacy.util import minibatch
from pipes.tokenizers import get_parser, parse_doc
from typing import List, Set, Dict, Tuple, Optional, Union, Callable, Generator

logger = logging.getLogger(__name__)


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
        logger.info(f'{f.__name__} took: {te - ts:2.4f} sec')
        return result

    return timed

def transform_texts(nlp,
                    batch_id,
                    corpus,
                    output_dir: str,
                    disable: Set[str] = None,
                    prefix: str = ''):
    """

    :param nlp:
    :param batch_id:
    :param corpus:
    :param output_dir:
    :param disable:
    :param prefix:
    :return:
    """
    out_path = Path(output_dir) / (
        f"{prefix + '.' if prefix else ''}{batch_id}.json")
    print("Processing batch", batch_id)

    with out_path.open("w", encoding="utf8") as f:
        doc_names, texts, metadata = zip(*corpus)
        for i, doc in enumerate(nlp.pipe(texts)):
            sents = list(parse_doc(doc, disable=disable))
            f.write(
                json.dumps(
                    {'name': str(doc_names[i]),
                     'metadata': metadata[i],
                     'sentences': sents}
                )
            )
            f.write("\n")
    print("Saved {} texts to JSON {}".format(len(texts), batch_id))


def dataloader(inputpath: str,
               primary_key: str = 'DOC_NAME',
               text_key: str = 'TEXT',
               preprocess: Callable = lambda x: x):
    """

    :param inputpath:
    :param preprocess:
    :return:
    """
    # directory or single file
    filelist =  glob.glob(inputpath + "/*.[tc]sv") \
        if os.path.isdir(inputpath) else [inputpath]

    for fpath in filelist:
        print(fpath)
        df = pd.read_csv(fpath, delimiter='\t', header=0, quotechar='"')
        for i, row in df.iterrows():
            doc_name = row[primary_key]
            text = row[text_key]
            text = text.replace('\\n', '\n').replace('\\t', '\t').replace('\\r', '\r')
            if not text.strip():
                logger.error(
                    f"Document {doc_name} contains no text -- skipping")
                continue
            # add any other columns as metadata
            metadata = {
                name:value for name,value in row.to_dict().items()
                if name not in [text_key]
            }
            yield (doc_name, preprocess(text), metadata)

def load_merge_terms(fpath:str, sep: str = '\t') -> Set[str]:
    terms = set()
    with open(fpath, 'r') as fp:
        for line in fp:
            terms.add(line.strip().split(sep)[0])
    return terms

@timeit
def main(args):

    merge_terms = load_merge_terms(args.merge_terms) \
        if args.merge_terms else {}

    nlp = get_parser(disable=args.disable.split(','),
                     merge_terms=merge_terms,
                     max_sent_len=args.max_sent_len)
    
    identity_preprocess = lambda x:x
    corpus = dataloader(
        args.inputdir,
        primary_key = args.primary_key,
        text_key = args.text_key,
        preprocess=identity_preprocess
    )

    partitions = minibatch(corpus, size=args.batch_size)
    executor = Parallel(n_jobs=args.n_procs,
                        backend="multiprocessing",
                        prefer="processes")
    do = delayed(partial(transform_texts, nlp))
    tasks = (do(i, batch, args.outputdir, args.disable, args.prefix) for
             i, batch in enumerate(partitions))
    executor(tasks)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-i", "--inputdir", type=str, default=None,
                           help="input directory")
    argparser.add_argument("-o", "--outputdir", type=str, default=None,
                           help="output directory")
    argparser.add_argument("-F", "--fmt", type=str, default="single",
                           help="document format (single|row)")
    argparser.add_argument("-p", "--prefix", type=str, default="",
                           help="json name prefix")
    argparser.add_argument("-n", "--n_procs", type=int, default=2,
                           help="number of processes")
    argparser.add_argument("-b", "--batch_size", type=int, default=1000,
                           help="batch size")
    argparser.add_argument("-d", "--disable", type=str,
                           default="ner,parser,tagger",
                           help="disable spaCy components")
    argparser.add_argument("--keep_whitespace", action='store_true',
                           help='retain whitespace tokens')

    argparser.add_argument("-m", "--max_sent_len", type=int, default=150,
                           help='Max sentence length (in words)')
    argparser.add_argument("-M", "--merge_terms", type=str, default=None,
                           help='Do not split lines spanning these phrases')

    argparser.add_argument("--quiet", action='store_true',
                           help="suppress logging")

    argparser.add_argument("--primary_key", type=str,
                           default="DOC_NAME",
                           help="primary document key")

    argparser.add_argument("--text_key", type=str,
                           default="TEXT",
                           help="text key column name")

    args = argparser.parse_args()

    if not args.quiet:
        FORMAT = '%(message)s'
        logging.basicConfig(format=FORMAT, stream=sys.stdout,
                            level=logging.INFO)

    logger.info(f'Python:      {sys.version}')
    for attrib in args.__dict__.keys():
        v = 'None' if not args.__dict__[attrib] else args.__dict__[attrib]
        logger.info("{:<15}: {:<10}".format(attrib, v))

    main(args)
