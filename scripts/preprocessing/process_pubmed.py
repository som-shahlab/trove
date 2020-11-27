import json
import spacy
from spacy.tokenizer import Tokenizer
from spacy.pipeline import SentenceSegmenter

def split_on_newline(doc):
    start = 0
    seen_newline = False
    for i, word in enumerate(doc):
        if seen_newline and word.text != '\n':
            yield doc[start:word.i]
            start = word.i
            seen_newline = False
        elif word.text == '\n':
            seen_newline = True
    if start < len(doc):
        yield doc[start:len(doc)]


def convert_annotation(item, etype=None):
    start, end, text, _type, cid = item
    return {
        "abs_char_start": start,
        "abs_char_end": end,
        "span": [[start, end]],
        "text": text,
        "etype": _type if not etype else etype,
        "cid": cid
    }


def create_splits(prefix,
                  indir,
                  outdir,
                  etype=None,
                  convert_etype=None):
    documents = json.load(open(f'{indir}/{prefix}.json', 'r'))
    splits = json.load(open(f'{indir}/{prefix}.splits.json', 'r'))
    entities = json.load(open(f'{indir}/{prefix}.tags.json', 'r'))

    for name in splits:
        data = []
        for doc_name in splits[name]:
            sentences = []
            for s in documents[doc_name]:
                sent = {
                    'i': s['position'],
                    'abs_char_offsets': s['abs_char_offsets'],
                    'words': s['words'],
                    'pos_tags': s['pos_tags']
                }
                sentences.append(sent)

            doc = {
                'name': doc_name,
                'sentences': sentences,
                'entities': [convert_annotation(item, etype=convert_etype) for item in entities[doc_name]]
            }
            if etype:
                doc['entities'] = [item for item in doc['entities'] if item['etype'] == etype]

            data.append(json.dumps(doc))

        outfpath = f'{outdir}/{name}.{prefix}.json' if not etype else f'{outdir}/{name}.{prefix}.{etype.lower()}.json'
        with open(outfpath, 'w') as fp:
            fp.write('\n'.join(data))
        print(outfpath, len(data))


def load_text_corpus(fpath, outdir, block_size=10000, skip={}):
    n = 0
    corpus = []
    stack = []
    with open(fpath, 'r') as fp:
        curr = None
        for i, line in enumerate(fp):
            row = line.strip().split("\t")
            doc_name, idx, abs_char_offsets, words = row

            if doc_name in skip:
                continue

            idx = int(idx)
            abs_char_offsets = list(map(int, abs_char_offsets.split(",")))

            doc = nlp(words)
            pos_tags = [tok.tag_ for tok in doc]
            words = words.split()

            if len(pos_tags) != len(words):
                print("ERROR", len(pos_tags), len(words))

            if curr and doc_name != curr:
                doc = {'name': curr, 'sentences': stack, 'entities': []}
                corpus.append(json.dumps(doc))
                stack = []

            curr = doc_name
            sent = {
                'i': idx,
                'abs_char_offsets': abs_char_offsets,
                'words': words,
                'pos_tags': pos_tags
            }
            stack.append(sent)

            if len(corpus) > 0 and len(corpus) % block_size == 0:
                outfpath = f'{outdir}/pubmed.{n}.{block_size}.json'
                with open(outfpath, 'w') as op:
                    op.write("\n".join(corpus))
                corpus = []
                n += 1
                print(outfpath)

        if corpus:
            outfpath = f'{outdir}/pubmed.{n}.{block_size}.json'
            with open(outfpath, 'w') as op:
                op.write("\n".join(corpus))



#
# Export NCBI + CDR datasets
#
prefix = 'cdr'
indir = '/Users/fries/Desktop/NER-Datasets-ALL/pubmed/cdr/dataset/'
outdir = '/users/fries/desktop/temp/'

create_splits(prefix, indir, outdir, etype='Disease')
create_splits(prefix, indir, outdir, etype='Chemical')

prefix = 'ncbi'
indir = '/Users/fries/Desktop/NER-Datasets-ALL/pubmed/ncbi/dataset/'
create_splits(prefix, indir, outdir, etype='Disease', convert_etype='Disease')


#
# Export Unlabeled PubMed Data
#
nlp = spacy.load('en')
nlp.tokenizer = Tokenizer(nlp.vocab)
sbd = SentenceSegmenter(nlp.vocab, strategy=split_on_newline)
nlp.add_pipe(sbd)

fpath = '/Users/fries/Desktop/pubmed/corpus/random_pubmed.100000.processed.txt'
outdir = '/users/fries/desktop/temp/'
load_text_corpus(fpath, outdir)