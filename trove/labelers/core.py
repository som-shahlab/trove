import itertools
import numpy as np
from scipy import sparse
from functools import partial
from toolz import partition_all
from joblib import Parallel, delayed
from abc import ABCMeta, abstractmethod


class Distributed:

    def __init__(self, num_workers=1, backend='multiprocessing'):
        self.client = Parallel(n_jobs=num_workers,
                               backend=backend,
                               prefer="processes")
        self.num_workers = num_workers
        print(self.client)


class SequenceLabelingServer(Distributed):

    @staticmethod
    def worker(lfs, data):
        return [[lf(x) for lf in lfs] for x in data]

    def apply(self, lfs, Xs, block_size=None):
        blocks = Xs
        if block_size is None:
            block_size = int(
                np.ceil(np.sum([len(x) for x in Xs]) / self.num_workers)
            )
            print(f'auto block size={block_size}')

        if block_size:
            blocks = list(
                partition_all(block_size, itertools.chain.from_iterable(Xs))
            )

        print(f"Partitioned into {len(blocks)} blocks, "
              f"{np.unique([len(x) for x in blocks])} sizes")

        do = delayed(partial(SequenceLabelingServer.worker, lfs))
        jobs = (do(batch) for batch in blocks)
        L = np.vstack(self.client(jobs))

        # merge matrix blocks
        Ls = []
        i = 0
        for n in [len(x) for x in Xs]:
            Ls.append(L[i:i + n].copy())
            i += n
        return Ls


class LabelingServer(Distributed):

    @staticmethod
    def worker(lfs, data):
        return sparse.csr_matrix(
            np.vstack([[lf(x) for lf in lfs] for x in data])
        )

    def apply(self, lfs, Xs, block_size=None):

        blocks = Xs
        if block_size is None:
            block_size = int(
                np.ceil(np.sum([len(x) for x in Xs]) / self.num_workers)
            )
            print(f'auto block size={block_size}')

        if block_size:
            blocks = list(
                partition_all(block_size, itertools.chain.from_iterable(Xs))
            )

        print(f"Partitioned into {len(blocks)} blocks, "
              f"{np.unique([len(x) for x in blocks])} sizes")

        do = delayed(partial(LabelingServer.worker, lfs))
        jobs = (do(batch) for batch in blocks)
        L = sparse.vstack(self.client(jobs))

        # merge matrix blocks
        Ls = []
        i = 0
        for n in [len(x) for x in Xs]:
            Ls.append(L[i:i + n].copy())
            i += n
        return Ls


class TaggerPipelineServer(Distributed):

    @staticmethod
    def worker(pipeline, corpus, ngrams=5):
        for doc in corpus:
            for name in pipeline:
                pipeline[name].tag(doc, ngrams=ngrams)
        return corpus

    def apply(self,
              pipeline,
              documents,
              block_size=None):

        items = itertools.chain.from_iterable(documents)

        if block_size is None:
            num_items = np.sum([len(x) for x in documents])
            block_size = int(np.ceil(num_items / self.num_workers))
            print(f'auto block size={block_size}')

        blocks = list(partition_all(block_size, items)) \
            if block_size else documents
        print(f"Partitioned into {len(blocks)} blocks, "
              f"{np.unique([len(x) for x in blocks])} sizes")

        do = delayed(partial(TaggerPipelineServer.worker, pipeline))
        jobs = (do(batch) for batch in blocks)
        results = list(itertools.chain.from_iterable(self.client(jobs)))

        i = 0
        items = []
        for n in [len(x) for x in documents]:
            items.append(results[i:i + n].copy())
            i += n
        return items
