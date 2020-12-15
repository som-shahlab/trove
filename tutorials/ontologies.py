import re
import pandas as pd


def apply_transforms(term, transforms):
    for tf in transforms:
        term = tf(term.strip())
        if not term:
            return None
    return term



 
class ChebiDatabase:
    
    _cfg = {
        'url': 'ftp://ftp.ebi.ac.uk/pub/databases/chebi/Flat_file_tab_delimited/names.tsv.gz'
    }
    _cache_path = "cache/chebi/"
    
    def __init__(self, cache_path, **kwargs):
        self.cache_path = cache_path
        self.df = self._load_terminologies(**kwargs)
        
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
    
        df = pd.read_csv('/users/fries/downloads/names.tsv', 
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
        