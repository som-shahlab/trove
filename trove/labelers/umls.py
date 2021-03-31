import os
import re
import errno
import shutil
import logging
import msgpack
import sqlite3
import hashlib
import pandas as pd
import urllib.request
from pathlib import Path
from zipfile import ZipFile
from collections import defaultdict

logger = logging.getLogger(__name__)

###############################################################################
#
# Unified Medical Language System (UMLS)
# 
###############################################################################

UMLS_VERS_MD5_CHECKSUMS = {
    'cf0699e5ca95d9eabc1a0b3a7f1bfda7': {'release': 'full', 'year':2020, 'version':'AB'},
    '9bca7282fa4ca6d0544e87628da67ddf': {'release': 'full', 'year':2018, 'version':'AA'},
    '69d2929e0902e7e42af0b2cb74d5005a': {'release': 'meta', 'year':2020, 'version':'AB'}
}

class UMLS:
    """
    Unified Medical Langauge System (UMLS) Metathesaurus
    Simple interface for loading UMLS terminologies and mapping
    terms to semantic types (TUI or CUI)
    https://www.nlm.nih.gov/research/umls/knowledge_sources/metathesaurus

    """
    cache_root = "~/.trove/umls"
    backend = 'pandas'

    def __init__(self, **kwargs):

        self.backend = kwargs['backend'] if 'backend' in kwargs \
            else UMLS.backend
        self.cache_path = UMLS.get_full_cache_path(
            kwargs['cache_path'] if 'cache_path' in kwargs
            else UMLS.cache_root
        )
        logger.info(f"cache_path= {self.cache_path}")
        logger.info(f"backend= {self.backend}")

        self._load_indices()
        self._apply_filters(**kwargs)

    @classmethod
    def config(cls, cache_root, backend):
        cls.cache_root = cache_root
        cls.backend = backend

    def _load_indices(self):

        if not UMLS.is_initalized(self.cache_path, self.backend):
            raise Exception("Error, UMLS not initialized.")

        # load mappings
        self.langs = {
            sab: lang for sab, lang, ssn in msgpack.load(
                open(f"{self.cache_path}/sabs.bin", 'rb')
            ).values()
        }
        self.tui_to_sty = msgpack.load(
            open(f"{self.cache_path}/tui_to_sty.bin", 'rb'))

    def _load_terminologies(self, filter_sabs, type_mapping='TUI'):

        if self.backend == 'pandas':
            df = pd.read_parquet(
                f"{self.cache_path}/concepts/",
                engine="pyarrow"
            ).groupby(['SAB'])

            for sab, data in df:
                if sab in filter_sabs:
                    continue
                yield sab, data.filter(items=['TERM', type_mapping]).values

        elif self.backend == 'sqlite':
            conn = sqlite3.connect(f"{self.cache_path}/umls.db")
            for sab in self.langs:
                if sab in filter_sabs:
                    continue
                sql = f'SELECT term, {type_mapping} ' \
                      f'FROM terminology WHERE sab="{sab}";'
                cursor = conn.execute(sql)
                rows = cursor.fetchall()
                yield sab, rows
            conn.close()

        else:
            raise Exception(f"Backend {self.backend} not recognized")

    def _apply_filters(self,
                       type_mapping='TUI',
                       min_char_len=2,
                       max_tok_len=6,
                       min_dict_size=500,
                       languages=None,
                       transforms=None,
                       filter_sabs=None,
                       filter_rgx=None,
                       stopwords=None):
        """
        Load concepts file and create transformed terminology dictionaries
        """
        # defaults
        filter_sabs = filter_sabs if filter_sabs else {'SNOMEDCT_VET'}
        languages = languages if languages else {}
        transforms = [] if not transforms else transforms
        stopwords = {} if not stopwords else stopwords
        filter_rgx = re.compile(filter_rgx) if filter_rgx else None

        if languages:
            filter_sabs.update({
                sab for sab in self.langs if self.langs[sab] not in languages
            })

        def include(t):
            return t and len(t) >= min_char_len and \
                   t.count(' ') <= max_tok_len - 1 and \
                   t not in stopwords and \
                   (filter_rgx and not filter_rgx.search(t))

        terminologies = {}

        for sab, data in self._load_terminologies(filter_sabs, type_mapping):

            # TODO enforce min constraint here or at SAB level
            #if len(data) < min_dict_size:
            #    continue

            if sab not in terminologies:
                terminologies[sab] = defaultdict(set)

            # TUI or CUI
            for term, cls_type in data:
                term = UMLS.apply_transforms(term, transforms)
                if include(term):
                    terminologies[sab][term].add(cls_type)

        self.terminologies = {
            sab: terminologies[sab] for sab in terminologies
            if len(terminologies[sab]) >= min_dict_size
        }

    @staticmethod
    def get_full_cache_path(root_dir=None):
        return os.path.expanduser(root_dir if root_dir else UMLS.cache_root)

    @staticmethod
    def apply_transforms(term, transforms):
        for tf in transforms:
            term = tf(term.strip())
            if not term:
                return None
        return term

    @staticmethod
    def init_sqlite_tables(fpath, dataframe):

        conn = sqlite3.connect(fpath)
        sql = """CREATE TABLE IF NOT EXISTS terminology (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        sab text NOT NULL,
                        tui text NOT NULL,
                        cui text NOT NULL,
                        term text NOT NULL
                    );"""
        conn.execute(sql)

        sql = """CREATE INDEX IF NOT EXISTS idx_sources 
                         ON terminology (sab);"""
        conn.execute(sql)

        sql = """CREATE INDEX IF NOT EXISTS idx_source_terms 
                 ON terminology (sab, tui);"""
        conn.execute(sql)

        rows = list(dataframe.itertuples())
        conn.executemany(
            "INSERT into terminology(sab, tui, cui, term) values (?,?,?,?)",
            rows)
        conn.commit()
        conn.close()

    @staticmethod
    def is_initalized(cache_root=None, backend=None):
        cache_path = UMLS.get_full_cache_path(cache_root)
        backend = backend if backend else UMLS.backend
        filelist = ['sabs.bin', 'tui_to_sty.bin', 'concepts']
        if backend == 'sqlite':
            filelist[-1] = 'umls.db'

        flags = [
            os.path.exists(f'{cache_path}/{fname}')
            for fname in filelist
        ]
        return flags.count(True) >= 3

    @staticmethod
    def reset(cache_root=None):
        cache_path = UMLS.get_full_cache_path(cache_root)
        if os.path.exists(cache_path):
            shutil.rmtree(cache_path)
            logger.info('UMLS cache reset')

    @staticmethod
    def init_from_dbconn():
        """
        TODO Implement this
        :return:
        """
        ...

    @staticmethod
    def init_from_nlm_zip(fpath,
                          outdir=None,
                          backend=None,
                          use_checksum=False,
                          keep_original_rrfs=False):
        """
        Install from NLM source zip file. This requires the 'metathesaurus'
        version (e.g., umls-2020AB-metathesaurus.zip) which contains the
        complete RRF file set.

        :param fpath:
        :return:
        """
        assert os.path.exists(fpath)
        # determine release
        # TODO: Checksum test is probably overkill
        # if use_checksum:
        #     checksum = hashlib.md5(open(fpath,'rb').read()).hexdigest()
        #     if checksum not in UMLS_VERS_MD5_CHECKSUMS:
        #         print("UMLS checksum not found, using string matching rule")
        #         use_checksum = False
        #     else:
        #         print(UMLS_VERS_MD5_CHECKSUMS[checksum])
        #         release = UMLS_VERS_MD5_CHECKSUMS[checksum]['release']
        if not use_checksum:
            release = 'full' if "full" in fpath.split('/')[-1] else 'meta'

        # TODO: Implement 'full' installation for archived UMLS zip files
        if release == 'full':
            msg = "Please use UMLS `metathesaurus` zip files, not `full`"
            raise Exception(msg)

        outdir = UMLS.get_full_cache_path(outdir)
        backend = backend if backend else UMLS.backend

        tmp = f'{outdir}/tmp'
        if not os.path.exists(tmp):
            os.makedirs(tmp)

        # extract files from zip
        deps = f"({'|'.join(['MRCONSO.RRF', 'MRSTY.RRF', 'MRSAB.RRF'])})"
        with ZipFile(fpath, 'r') as zipfile:
            for fname in zipfile.namelist():
                if not re.search(deps, fname):
                    continue
                outfile = Path(f'{tmp}/{os.path.basename(fname)}')
                if os.path.exists(outfile):
                    continue
                outfile.write_bytes(zipfile.read(fname))

        # init from RRFs
        UMLS.init_from_rrfs(tmp, outdir, backend)
        if not keep_original_rrfs:
            shutil.rmtree(tmp)
            logger.info(f"Deleted temporary UMLS files at {tmp}")

    @staticmethod
    def init_from_rrfs(indir, outdir=None, backend=None):
        """
        Initialize UMLS from Rich Release Format (RRF) files
        see https://www.ncbi.nlm.nih.gov/books/NBK9685/

        :param indir:
        :param outdir:
        :return:
        """
        outdir = UMLS.get_full_cache_path(outdir)
        backend = backend if backend else UMLS.backend
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        # validate that the UMLS source REFs are provided
        for fname in ['MRCONSO.RRF', 'MRSTY.RRF', 'MRSAB.RRF']:
            if not os.path.exists(f"{indir}/{fname}"):
                raise FileNotFoundError(
                    errno.ENOENT,
                    os.strerror(errno.ENOENT),
                    fname
                )

        # Source terminologies - MRSAB.RRF
        sabs = {}
        with open(f'{indir}/MRSAB.RRF', 'r') as fp:
            for line in fp:
                row = line.strip('').split('|')
                # ignore RSAB version
                rsab, _, lat, ssn = row[3], row[6], row[19], row[23]
                if rsab in sabs:
                    continue
                sabs[rsab] = (rsab, lat, ssn)

        with open(f'{outdir}/sabs.bin', 'wb') as fp:
            fp.write(msgpack.dumps(sabs))

        # Concept Unique ID to Semantic Type mappings - MRSTY.RRF
        tui_to_sty = {}
        cui_to_tui = defaultdict(set)
        with open(f'{indir}/MRSTY.RRF', 'r') as fp:
            for line in fp:
                row = line.strip('').split('|')
                cui, tui, sty = row[0], row[1], row[3]
                cui_to_tui[cui].add(tui)
                tui_to_sty[tui] = sty

        with open(f'{outdir}/tui_to_sty.bin', 'wb') as fp:
            fp.write(msgpack.dumps(tui_to_sty))

        # MRCONSO.RRF
        with open(f'{indir}/MRCONSO.RRF', 'r') as fp, open(
                f'{outdir}/concepts.tsv', 'w') as op:
            op.write('SAB\tTUI\tCUI\tTERM\n')
            for line in fp:
                row = line.strip().split('|')
                cui, sab, term = row[0], row[11], row[14]
                if term.strip() is None:
                    continue
                for tui in cui_to_tui[cui]:
                    op.write(f'{sab}\t{tui}\t{cui}\t{term}\n')

        df = pd.read_csv(
            f'{outdir}/concepts.tsv',
            sep='\t',
            header=0,
            quotechar=None,
            quoting=3,
            index_col=0 if backend == 'sqlite' else None,
            na_filter=False,
            dtype={
                'SAB': 'object',
                'TUI': 'object',
                'CUI': 'object',
                'TERM': 'object'
            }
        )

        if backend == 'pandas':
            df.to_parquet(f'{outdir}/concepts', partition_cols=['SAB'])
        elif backend == 'sqlite':
            UMLS.init_sqlite_tables(f'{outdir}/umls.db', df)
        # cleanup temp files
        os.remove(f'{outdir}/concepts.tsv')


class SemanticGroups:
    """
    Load the UMLS Semantic groups
    """
    cache_root = "~/.trove/semantic_groups"

    def __init__(self, cache_path=None):
        cache_path = cache_path if cache_path else SemanticGroups.cache_root
        cache_path = os.path.expanduser(cache_path)
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        if not os.path.exists(f"{cache_path}/SemGroups.txt"):
            self._init_data(cache_path)
        self._df = pd.read_csv(f"{cache_path}/SemGroups.txt", sep="|",
                               names=['GRP', 'GRP_STR', 'TUI', 'STR'])

    def _init_data(self, cache_path):
        try:
            url = 'https://semanticnetwork.nlm.nih.gov/download/SemGroups.txt'
            urllib.request.urlretrieve(url, f"{cache_path}/SemGroups.txt")
            logger.info(f"Downloaded {url}")
        except Exception as e:
            logger.error(f"{e} could not download {url}")

    @property
    def groups(self):
        d = defaultdict(set)
        for row in self._df.itertuples():
            d[row.GRP].add(row.TUI)
        return dict(d)

    @property
    def types(self):
        return {row.TUI: row.STR for row in self._df.itertuples()}
