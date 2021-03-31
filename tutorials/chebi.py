import re
import os
import gzip
import pandas as pd
import urllib.request

def download(url, outfpath):
    print(f'downloading {url}')
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-agent", "Mozilla/5.0")]
    urllib.request.install_opener(opener)
    urllib.request.urlretrieve(url, f"{outfpath}/{os.path.basename(url)}")

def apply_transforms(term, transforms):
    for tf in transforms:
        term = tf(term.strip())
        if not term:
            return None
    return term

class ChebiDatabase:
    
    cfg = {
        'url': 'ftp://ftp.ebi.ac.uk/pub/databases/chebi/Flat_file_tab_delimited/names.tsv.gz'
    }
    cache_root = "~/.trove/chebi/"
    
    def __init__(self, cache_path=None, **kwargs):
        self.cache_path = cache_path if cache_path else ChebiDatabase.cache_root
        self.cache_path = os.path.expanduser(self.cache_path)
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
        if not os.path.exists(f'{self.cache_path}/names.tsv'):
            download(ChebiDatabase.cfg['url'], self.cache_path)
            with gzip.open(f'{self.cache_path}/names.tsv.gz', 'rb') as fp:
                with open(f'{self.cache_path}/names.tsv', 'wb') as op:
                    op.write(fp.read())

        self.df = self._load_terminologies(**kwargs)

    @classmethod
    def config(cls, cache_root):
        cls.cache_root = cache_root

    def terms(self, filter_sources=None):
        
        filter_sources = filter_sources if filter_sources else {}
        terms = set()
        for source in self.terminologies:
            if source in filter_sources:
                continue
            terms = terms.union(self.terminologies[source])
        return terms
        
    def _load_terminologies(self,
                            min_char_len=2,
                            max_tok_len=100,
                            min_dict_size=1,
                            languages=None,
                            transforms=None,
                            filter_sources=None,
                            filter_rgx=None,
                            stopwords=None):
           
        # defaults
        languages = languages if languages else {}
        transforms = [] if not transforms else transforms
        filter_sources = filter_sources if filter_sources else {}
        filter_rgx = re.compile(filter_rgx) if filter_rgx else None
        stopwords = {} if not stopwords else stopwords

        def include(t):
            return t and len(t) >= min_char_len and \
                   t.count(' ') <= max_tok_len - 1 and \
                   t not in stopwords and \
                   (filter_rgx and not filter_rgx.search(t))
    
        df = pd.read_csv(f'{self.cache_path}/names.tsv',
                         sep='\t', 
                         na_filter=False, 
                         dtype={'NAME':'object', 'COMPOUND_ID':'object'})
        
        self.terminologies = {}
        if languages:
            df = df[df.LANGUAGE.isin(languages)]
            
        for source, data in df.groupby(['SOURCE']):
            if source in filter_sources or len(data) < min_dict_size:
                continue
            self.terminologies[source] = set()
            
            for term in data.NAME:
                term = apply_transforms(term, transforms)
                if include(term):
                    self.terminologies[source].add(term)
        self.data = df
