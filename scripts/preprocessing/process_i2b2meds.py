import sys
sys.path.insert(0,'/Users/fries/Code/inkNER/')

import json
import gzip
from trove.data.datasets.i2b2meds import i2b2Meds2009Dataset
from trove.data.dataloaders import dataloader


def doc_to_json(doc, entities=[]):
    d = {'name': doc.name, 'sentences': [], 'entities': []}
    for sent in doc.sentences:
        sent = {attrib: sent.__dict__[attrib]
                for attrib in sent.__dict__ if attrib not in ['document']}
        d['sentences'].append(sent)
    for item in entities:
        d['entities'].append(item.__dict__)
    return json.dumps(d)


def dump_json(dataset, annotations, fpath):
    print(fpath)
    with open(fpath, 'w') as fp:
        for doc in dataset:
            entities = annotations[doc.name] if doc.name in annotations else []
            data = doc_to_json(doc, entities)
            fp.write(f'{data}\n')


def load_json_docs(fpath):
    doc_idx = {}
    fopen = gzip.open if fpath.split(".")[-1] == 'gz' else open
    with fopen(fpath, 'rb') as fp:
        for line in fp:
            d = json.loads(line)
            doc_idx[d['name']] = d
    return doc_idx

root_dir = f'/Users/fries/Desktop/NER-Datasets-ALL/ehr/i2b2meds/'
i2b2 = i2b2Meds2009Dataset(anno_dir=f'{root_dir}/raw/annotations/',
                           doc_dir=f'{root_dir}/raw/docs/',
                           entity_types=['drug'])

# divide into splits
splits = json.load(open(f'{root_dir}/dataset/i2b2meds.splits.seed_1234.json','r'))
# load parsed documents
documents = list(dataloader([f'{root_dir}/dataset/docs/i2b2meds.0.json']))

dataset = {name:[doc for doc in documents if doc.name in splits[name]] for name in splits}

for name in dataset:
    print(name, len(dataset[name]))


outdir = '/Users/fries/Desktop/debug/'
etype = 'drug'
corpus = 'i2b2meds'

for name in ['train','dev','test']:
    outfpath = f'{outdir}/{name}.{corpus}.{etype}.json'
    dump_json(dataset[name], i2b2.annotations, outfpath)


