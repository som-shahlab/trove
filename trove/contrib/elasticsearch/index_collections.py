import sys
import glob
import json
import argparse
import numpy as np
import multiprocessing
from elasticsearch import Elasticsearch, helpers
from elasticsearch.helpers import parallel_bulk, bulk
from trove.dataloaders import DocumentLoader


def doc_loader(iterable, index_name):
    for doc in iterable:
        yield {
            '_op_type': 'index',
            '_index': index_name,
            '_source': doc
        }


def bulk_index(client, index_name, documents, n_threads=1, chunk_size=500):

    succeeded, failed = 0, 0

    if n_threads > 1:
        bulk_loader = parallel_bulk(client, doc_loader(documents, index_name),
                                    thread_count=n_threads,
                                    chunk_size=chunk_size)
        # returns result per document
        for success, info in bulk_loader:
            if not success:
                print(f'Doc failed {info}')
                failed += 1
            else:
                succeeded += 1
                #if succeeded % 1000000 == 0:
                #    print(f'Completed {succeeded}')
    else:
        bulk_loader = bulk(client, doc_loader(documents, index_name),
                           chunk_size=chunk_size)
        # returns single summary
        succeeded, info = list(bulk_loader)

    print(f"succeeded:{succeeded} failed:{failed}")


def document_loader(root_dir):
    for doc in DocumentLoader(root_dir):
        yield {
            'name': doc.name,
            'mrn': doc.props['MRN'],
            'encounter': doc.props['ENCOUNTER'],
            'created_at': doc.props['CREATED_AT'],
            'modified_at': doc.props['MODIFIED_AT'],
            'note_type': "UNK",
            'text': doc.text,
            'n_sents': len(doc.sentences),
        }


def sentence_loader(root_dir):
    for doc in DocumentLoader(root_dir):
        for sent in doc.sentences:
            yield {
                'doc_name': doc.name,
                'i': sent.i,
                'words': sent.words,
                'abs_char_offsets': sent.abs_char_offsets,
                'pos_tags': sent.pos_tags,
                'text': sent.text
            }


def entity_loader(filelist):
    names = ['doc_name', 'modified_at', 'entity_type', 'text',
             'abs_char_start',
             'abs_char_end', 'polarity', 'hypothetical', 'doc_rela_time',
             'section', 'subject', 'tdelta']
    dtypes = [str, str, str, str, int, int, str, str, str, str, str, str]

    for fpath in filelist:
        with open(fpath, 'r') as fp:
            for i, line in enumerate(fp):
                row = line.strip().split("\t")
                if i == 0:
                    continue
                row = [f(x) if x != 'NULL' else None for f, x in
                       zip(dtypes, row)]
                row = dict(zip(names, row))
                row['source'] = 'trove'
                del row['modified_at']
                yield row


def worker_process(host, port, index_name, fpaths, loader, n_threads, chunk_size):

    client = Elasticsearch([{'host': host, 'port': port}])

    bulk_index(client,
               index_name,
               loader(fpaths),
               n_threads=n_threads,
               chunk_size=chunk_size)


def main(args):

    # load JSON mappings
    mappings = json.load(open('mappings.json'))

    client = Elasticsearch([{'host': args.host, 'port': args.port}])

    # clear index
    if args.clear_index and client.indices.exists(args.index_name):
        print(f"Deleting {args.index_name} index")
        res = client.indices.delete(index=args.index_name)
        print(res)

    if not client.indices.exists(args.index_name):
        print(f"Created {args.index_name}")

        res = client.indices.create(
            index=args.index_name,
            body=mappings[args.index_name],
            ignore=400
        )
        print(res)

    # spin up index workers
    filelist = glob.glob(f"{args.inputdir}/*")
    if len(filelist) == 0:
        print('No files found, exiting')
        return

    blocks = np.array_split(filelist, args.n_procs)

    if args.index_name == 'documents':
        loader = document_loader
    elif args.index_name == 'sentences':
        loader = sentence_loader
    elif args.index_name == "entities":
        loader = entity_loader
    else:
        print(f"{args.index_name} not found")
        return

    try:
        workers = []
        for i in range(args.n_procs):
            print(f"worker_{i}: Files {len(blocks[i])}")
            worker = multiprocessing.Process(target=worker_process,
                                             args=(args.host,
                                                   args.port,
                                                   args.index_name,
                                                   blocks[i],
                                                   loader,
                                                   args.n_bulk_threads,
                                                   args.chunk_size))
            workers.append(worker)
            worker.start()

    finally:
        for w in workers:
            w.join()

        print('Done!')

if __name__=="__main__":

    choices = ['documents', 'sentences', 'entities']

    parser = argparse.ArgumentParser()
    parser.add_argument("--index_name", type=str, choices=choices, default=None, required=True)
    parser.add_argument("--inputdir", type=str, default=None, required=True)
    parser.add_argument("--clear_index", action="store_true")
    parser.add_argument("--n_bulk_threads", type=int, default=4)
    parser.add_argument("--n_procs", type=int, default=4)
    parser.add_argument("--chunk_size", type=int, default=1000)

    parser.add_argument("--port", type=int, default=9200)
    parser.add_argument("--host", type=str, default="localhost")
    args = parser.parse_args()

    print(json.dumps(vars(args), indent=4))
    main(args)