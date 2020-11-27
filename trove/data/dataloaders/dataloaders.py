import glob
import gzip
import json
from .contexts import Document, Sentence

def parse_doc(d) -> Document:
    """Convert JSON into container objects. Most time is spent loading JSON.
    Transforming to Document/Sentence objects comes at ~13% overhead.

    Parameters
    ----------
    d
        dictionary of document kwargs

    Returns
    -------

    """
    sents = [Sentence(**s) for s in d['sentences']]
    doc = Document(d['name'], sents)
    if 'metadata' in d:
        for key,value in d['metadata'].items():
            doc.props[key] = value
    return doc


def dataloader(filelist):
    """
    Load compressed JSON files
    """
    documents = []
    for fpath in filelist:
        fopen = gzip.open if fpath.split(".")[-1] == 'gz' else open
        with fopen(fpath,'rb') as fp:
            for line in fp:
                doc = parse_doc(json.loads(line))
                documents.append(doc)
    return documents

def init_dataset(splits, corpus):
    """
    Assign loaded documents to splits defined by a JSON file provided at `fpath`
    """
    dataset = {name:[] for name in ['train', 'dev', 'test', 'unlabeled']}
    for doc in corpus:
        if doc.name in splits:
            dataset[splits[doc.name]].append(doc)
        else:
            dataset['unlabeled'].append(doc)
    return dataset


def load_dataset(doc_root, splits, preprocess=None):
    """
    :param: preprocess apply a function to the entire corpus
    """
    filelist = glob.glob(f"{doc_root}/*")
    assert len(filelist) > 0
    corpus = dataloader(filelist)
    if preprocess:
        preprocess(corpus) 
        
    # create dataset splits
    if type(splits) is str:
        splits = json.load(open(splits,'r'))
        splits = {doc_name:s for s in splits for doc_name in splits[s]}
        
    dataset = init_dataset(splits, corpus)
    for name in dataset:
        print(f'{name.upper():<10}: {len(dataset[name]):>6}')
    return dataset

