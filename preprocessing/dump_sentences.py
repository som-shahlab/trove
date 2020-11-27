import sys
sys.path.insert(0,'/data4/jfries/code/ehr-rwe/')

import glob
import gzip
import json
from rwe.contexts import Document, Sentence

def parse_doc(d) -> Document:
    sents = [Sentence(**s) for s in d['sentences']]
    doc = Document(d['name'], sents)
    if 'metadata' in d:
        for key,value in d['metadata'].items():
            doc.props[key] = value
    return doc

def dataloader(filelist):
    """
    Load compresssed JSON files
    """
    documents = []
    for fpath in filelist:
        fopen = gzip.open if fpath.split(".")[-1] == 'gz' else open
        with fopen(fpath,'rb') as fp:
            for line in fp:
                yield parse_doc(json.loads(line))


filelist = glob.glob(f"{sys.argv[1]}/*.json")
corpus = dataloader(filelist)

for doc in corpus:
    for s in doc.sentences:
        print(s.text)
    break