import sys
sys.path.insert(0,'/Users/fries/Code/inkNER/')
import json
import argparse
from trove.data.datasets.mimic import (
    synthetic_mimic_dates, fix_mimic_blinding
)
from trove.data.dataloaders import dataloader


def doc_to_json(doc):
    d = {'name': doc.name, 'sentences': []}
    for sent in doc.sentences:
        sent = {attrib: sent.__dict__[attrib] \
                for attrib in sent.__dict__ if attrib not in ['document']}
        d['sentences'].append(sent)
    return json.dumps(d)


def dump_json(dataset, fpath):
    print(fpath)
    with open(fpath, 'w') as fp:
        for doc in dataset:
            data = doc_to_json(doc)
            fp.write(f'{data}\n')


def main(args):

    documents = list(dataloader([args.infile]))

    # fix MIMIC date tokens
    synthetic_mimic_dates(documents)
    fix_mimic_blinding(documents)

    dump_json(documents, args.outfile)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--infile", type=str, required=True)
    argparser.add_argument("--outfile", type=str, required=True)
    args = argparser.parse_args()

    main(args)
