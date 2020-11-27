import json
import gzip
import glob
import itertools
from trove.data.dataloaders import dataloader
from trove.data.datasets.clef import Clef2014Dataset
from trove.data.dataloaders.contexts import Annotation
from trove.data.datasets.mimic import synthetic_mimic_dates, fix_mimic_blinding


def doc_to_json(doc, entities=[]):
    d = {'name': doc.name, 'sentences': [], 'entities': []}
    for sent in doc.sentences:
        sent = {attrib: sent.__dict__[attrib] for attrib in sent.__dict__ if attrib not in ['document']}
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


def load_clef_entities(annotations, target='disorder'):
    annos = {}
    for item in annotations:
        if target == 'disorder':
            entity = item.disorder
        elif target == 'bodylocation':
            entity = item.bodylocation
            if not entity:
                continue
        cui = list(entity.attribs.keys())[0]
        if entity.doc_name not in annos:
            annos[entity.doc_name] = []
        annos[entity.doc_name].append(Annotation(entity.doc_name, entity.span, entity.type, entity.text, cui))
    return annos


root_dir = '/Users/fries/Desktop/'
data_root = f"{root_dir}/ShareCLEF_2014/task2/corpus_v2/"

clef_dataset = Clef2014Dataset(anno_dir=f'{data_root}annos/',
                               doc_dir=f'{data_root}docs/',
                               splits_filepath='/Users/fries/Desktop/CLEF-FINAL/10k-subset/clef.splits.v2.json')

splits = json.load(open('/Users/fries/Desktop/CLEF-FINAL/10k-subset/clef.splits.v2.json', 'r'))

# gold annotations
annotations = list(itertools.chain.from_iterable(clef_dataset.get_annotations().values()))
print(len(annotations))

disorders = load_clef_entities(annotations, target='disorder')
bodylocations = load_clef_entities(annotations, target='bodylocation')

# load parsed documents
documents = list(dataloader(['/Users/fries/Desktop/CLEF-FINAL/processed/json-pos/clef.0.json']))

# fix MIMIC date tokens
synthetic_mimic_dates(documents)
fix_mimic_blinding(documents)

# divide into splits
dataset = {name: [doc for doc in documents if doc.name in splits[name]] for name in splits}

# outdir = '/Users/fries/Desktop/NER-Datasets-ALL/ehr/shareclef/preprocessed/'
outdir = '/Users/fries/Desktop/debug/'
# etype = 'disorder'
etype = 'bodylocation'

for name in ['train', 'dev', 'test']:
    outfpath = f'{outdir}/{name}.shareclef.{etype}.json'
    dump_json(dataset[name], disorders if etype == 'disorder' else bodylocations, outfpath)

#
# unlabeled MIMIC data
#
filelist = glob.glob("/Users/fries/Desktop/NER-Datasets-ALL/ehr/mimic3/preprocessed/parsed/*")

name = 'unlabeled'
for fpath in filelist:
    block_name = fpath.split("/")[-1]
    documents = list(dataloader([fpath]))
    synthetic_mimic_dates(documents)
    fix_mimic_blinding(documents)

    outfpath = f'{outdir}/{name}.{block_name}'
    print(outfpath)
    dump_json(documents, {}, outfpath)
