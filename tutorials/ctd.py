import requests
import re
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from abc import ABCMeta, abstractmethod

from tqdm import tqdm
import urllib.request


class ProgressBar(tqdm):
    """
    Based on https://gist.github.com/leimao/37ff6e990b3226c2c9670a2cd1e4a6f5
    """
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize

def download(url, save_dir):
    fname = url.split('/')[-1]
    opener = urllib.request.build_opener()
    opener.addheaders = [("User-agent", "Mozilla/5.0")]
    urllib.request.install_opener(opener)
        
    with ProgressBar(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=fname) as t:
        urllib.request.urlretrieve(url, filename=os.path.join(save_dir, fname), reporthook=t.update_to)

def apply_transforms(term, transforms):
    for tf in transforms:
        term = tf(term.strip())
        if not term:
            return None
    return term
        
        
class KnowledgeBase(metaclass=ABCMeta):
    """
    We use Knowledge Base to loosely refer to a structured resource
    that contains terminology information. We are interested in the 
    following properties:
    
    - term typing
    - synonomy
    
    When source information is available, we store the above info mapped to source.
    
    """
    _cache_path = "cache/"
    
    def __init__(self, cache_path, files, force_download=False):
        
        self.cache_path = cache_path
        self.files = files
        
        if not self._check_cache() or force_download:
            self._download()

    def _download(self):
        
        for fname,url in self.files.items():
            download(url, self.cache_path)
  
    def _check_cache(self):
        """
        Confirm all file dependencies exist in the cache.
        """
        if not os.path.exists(self.cache_path):
            os.makedirs(self.cache_path)
            return False
        
        for fname in self.files:
            if not os.path.exists(f"{self.cache_path}{fname}"):
                return False
        return True
        
    @abstractmethod
    def name(self):
        ...
        
    @abstractmethod
    def manifest(self):
        ...    
        
    @abstractmethod
    def _load(self, **kwargs):
        ...
    
    @abstractmethod
    def get_source_terms(self):
        ...
    
    @abstractmethod
    def get_source_synsets(self):
        ...
    


class CtdDatabase(KnowledgeBase):

    cache_root = "~/.trove/"

    def __init__(self, cache_path=None, **kwargs):

        cache_root = cache_path if cache_path else CtdDatabase.cache_root
        cache_root = os.path.expanduser(cache_root)
        force_download = kwargs['force_download'] if 'force_download' in kwargs else False
        
        super().__init__(
            cache_path = f"{cache_root}{self.name}/",
            files = self.manifest,
            force_download = force_download
        )
        
        self.terms = {}
        self.data = self._load()
        
        for name,key in {'disease':'DiseaseName', 'chemical':'ChemicalName'}.items():
            self.terms[name] = self._collapse_terms(self.data[name], key)
            self.terms[name] = self._transform_terminologies(self.terms[name], **kwargs)
            
        # TODO
        self.synset = {}
        
    @property
    def name(self):
        return 'ctd'
        
    @property
    def manifest(self):
        return {
            'CTD_diseases.csv.gz' : 'http://ctdbase.org/reports/CTD_diseases.csv.gz',
            'CTD_chemicals.csv.gz' : 'http://ctdbase.org/reports/CTD_chemicals.csv.gz'
        }      
    
    def _collapse_terms(self, df, key):
        """
        CTD includes ID: terms -> synonyms. We just collapse 
        all terms into a single entity dictionary.
        """
        terms = set()
        for row in df.itertuples():
            if not pd.isnull(getattr(row, key)):
                terms.add(getattr(row, key))
            if not pd.isnull(row.Synonyms):
                for term in row.Synonyms.split("|"):
                    terms.add(term)
        return terms

    
    def _load_disease_data(self):
        
        columns = [
            'DiseaseName',
            'DiseaseID',
            'AltDiseaseIDs',
            'Definition',
            'ParentIDs',
            'TreeNumbers',
            'ParentTreeNumbers',
            'Synonyms',
            'SlimMappings'
        ]
        
        fpath = f"{self.cache_path}/CTD_diseases.csv.gz"
        return pd.read_csv(
            fpath, 
            comment='#', 
            sep=',', 
            names=columns,
            dtype=str
        )
    
    def _load_chemical_data(self):
        
        columns = [
            'ChemicalName',
            'ChemicalID',
            'CasRN',
            'Definition',
            'ParentIDs',
            'TreeNumbers',
            'ParentTreeNumbers',
            'Synonyms'
        ]
        
        fpath = f"{self.cache_path}/CTD_chemicals.csv.gz"
        return pd.read_csv(
            fpath, 
            comment='#', 
            sep=',', 
            names=columns,
            dtype=str
        )
    
    def _transform_terminologies(self,
                            terms,
                            min_char_len=2,
                            max_tok_len=100,
                            transforms=None,
                            filter_rgx=r'''^[0-9]$''',
                            stopwords=None,
                            **kwargs):
        
        transforms = [] if not transforms else transforms
        filter_rgx = re.compile(filter_rgx) if filter_rgx else None
        stopwords = {} if not stopwords else stopwords

        def include(t):
            return t and len(t) >= min_char_len and \
                   t.count(' ') <= max_tok_len - 1 and \
                   t not in stopwords and \
                   (filter_rgx and not filter_rgx.search(t))
    
        tmp = set()
        for term in terms:
            term = apply_transforms(term, transforms)
            if include(term):
                tmp.add(term)
        return tmp
    
    def _load(self):
        
        return {
            "disease" : self._load_disease_data(),
            "chemical" : self._load_chemical_data()
        }
      
    def get_source_terms(self, source):
        assert source in self.data
        return self.terms[source]
    
    def get_source_synsets(self, source):
        pass
        



