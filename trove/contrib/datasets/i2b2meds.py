import re
import glob
from typing import List, Dict
from ..dataloaders.contexts import Annotation


class i2b2Meds2009Dataset(object):
    """
    i2b2 2009 Medication Extraction Challenge Dataset
    
    We update the format of this dataset as follows:
    - Entities are labeled by document-level character offsets
      (vs. line/word offsets)
    - Tokenization is not restricted by whitespace
    - Gold entities do not cross sentence boundaries (in the original dataset
      this was defined by newlines, not actual sentence boundaries).

    Dataset Summary:
        Docs:    250
        Entites: 8480  Discontinuous Entities: 11

    TODO: 
    - Include relational information across entities
    - Add annotations for medications denoted as allergies. The original
      challenge *did not* include medications listed as allergies. This
      causes many (incorrect) false postives when tagging for drug concepts.

    Example Usage:

        root_dir = '/Users/fries/Desktop/i2b2-FINAL/release/'
        doc_root = '/Users/fries/Desktop/i2b2-FINAL/processed/txt/'

        i2b2 = i2b2Meds2009Dataset(anno_dir=f'{root_dir}annos/',
                                   doc_dir=doc_root)

        for doc_name in i2b2.annotations:
            doc_entities = i2b2.annotations[doc_name]

    """
    def __init__(self,
                 anno_dir: str,
                 doc_dir: str,
                 entity_types: List[str] = None,
                 verbose: bool = True):

        self.verbose = verbose
        self.entity_types = entity_types
        docs = glob.glob(f'{doc_dir}/*')
        annos = glob.glob(f'{anno_dir}/*')
        print(f'Docs:  {len(docs)}')
        print(f'Annos: {len(annos)}')

        anno_map = {fname.split('/')[-1].split(".")[0]: fname for fname in annos}
        doc_map = {fname.split('/')[-1].split(".")[0]: fname for fname in docs}

        files = [(doc_map[key], anno_map[key]) for key in anno_map]
        self.annotations = self.load_annotations(files)

    def get_offset_index(self, text: str) -> Dict[int, Dict[int, int]]:
        """
        Create a index that maps from i2b2 annotation offset
        notation to absolute char offsets.
        """
        sentences = []
        curr_char_offset = 0
        offset_index = {}
        for sent_i, sent in enumerate(text.split('\n')):
            offset_index[sent_i] = {}
            tokens = re.split('(\s)', sent)
            token_i = 0
            for t in tokens:
                if t.strip():
                    offset_index[sent_i][token_i] = [curr_char_offset, len(t)]
                    token_i += 1
                curr_char_offset += len(t)
            # offset for newline char
            curr_char_offset += 1
            sentences.append(sent)
        return offset_index, sentences

    def get_anno_span(self, anno: str):
        """
        Parse annotation row
        TODO: Add linkage across relations (Drug - Dose - Mode etc)

        """
        rgx = r'''(m|do|mo|f|du|r|ln)="(nm|list|narrative|.+?)"(\s+((\d+[:]\d+) (\d+[:]\d+)[,]*){1,})*'''
        etypes = {
            'm' : 'drug',
            'do': 'dose',
            'mo': 'mode',
            'f' : 'freq',
            'du': 'duration',
            'r' : 'reason'
        }

        entities = []
        for item in anno.split("||"):
            m = re.search(rgx, item, re.I)
            etype = m.group(1)
            text = m.group(2)
            # skip these designations
            if text in ['nm', 'list', 'narrative']:
                continue
            spans = []
            for s in m.group(3).split(","):
                m = re.search(r'''(\d+[:]\d+) (\d+[:]\d+)''', s, re.I)
                start = list(map(int, m.group(1).split(":")))
                end = list(map(int, m.group(2).split(":")))
                start = tuple((start[0] - 1, start[1]))
                end = tuple((end[0] - 1, end[1]))
                spans.append((start, end))
            entities.append((etypes[etype], text, tuple(spans)))
        return entities

    def parse_anno_markup(self, fpath: str):
        entities = []
        with open(fpath, 'r') as fp:
            for line in fp:
                for item in line.strip().split('||'):
                    anno = self.get_anno_span(item)
                    entities.extend(anno)
        # generate unique spans only
        return sorted(list(set(entities)), key=lambda x: x[2], reverse=0)

    def apply_anno_fixes(self, doc_name:str,
                         text:str,
                         etype:str,
                         spans):
        """
        Clean up some gold annotations.

        :param doc_name:
        :param text:
        :param etype:
        :param spans:
        :return:
        """
        # document-specific fixes
        fixes = {
            ('413813', ((3791, 3804),), 'MEDS:Protonix'): ('413813', ((3796, 3804),), 'Protonix'),
            ('498500', ((3497, 3511),), 'analgesics.She'): ('498500', ((3497, 3507),), 'analgesics'),
            ('701307', ((3079, 3087),), 'meds:ASA'): ('701307', ((3084, 3087),), 'ASA'),
            ('758638', ((5298, 5314),), 'Norvasc.Consider'): ('771801', ((5298, 5305),), 'Norvasc'),
            ('767633', ((6896, 6909),), 'hrs.Toprol XL'): ('767633', ((6900, 6909),), 'Toprol XL'),
            ('771801', ((3253, 3262),), 'celondin-'): ('771801', ((3253, 3261),), 'celondin')
        }

        span, mention = zip(*spans)
        span = tuple(map(tuple, span))
        mention = '...'.join(mention)
        key = (doc_name, span, mention)
        if key in fixes:
            _, span, mention = fixes[key]
            # santity check fixes
            s = span[0]
            if mention != text[s[0]:s[1]]:
                print(f'Fix ERROR {mention} != {text[s[0]:s[1]]}')
            return [[span[0], mention]]

        # fix tokens by stripping any punctuation
        stopwords = []
        if etype in ['drug', 'duration', 'reason']:
            tmp = []
            for ((start, end), mention) in spans:
                if mention.lower() not in stopwords and re.search(r'''[.,:;]$''', mention, re.I):
                    end -= 1
                    mention = mention[:-1]
                tmp.append([(start, end), mention])
            spans = tmp

        return spans

    def _parse(self, doc_fpath: str, anno_fpath: str):
        """
        Create Annotation objects for all annotated spans.

        :param doc_fpath:
        :param anno_fpath:
        :return:
        """
        doc_name = doc_fpath.split("/")[-1]
        text = open(doc_fpath, 'r').read()
        markup = self.parse_anno_markup(anno_fpath)
        token_offset_index, sentences = self.get_offset_index(text)
        text = '\n'.join(sentences)

        annotations = []
        for item in markup:
            etype, term, span = item
            # filter entity types
            if self.entity_types and etype not in self.entity_types:
                continue

            try:
                anno_spans = []
                for start, end in span:
                    start_sent_i, start_tok_i = start
                    end_sent_i, end_tok_i = end
                    i, len_i = token_offset_index[start_sent_i][start_tok_i]
                    j, len_j = token_offset_index[end_sent_i][end_tok_i]
                    anno_spans.append([[i, j + len_j], text[i:j + len_j]])

                anno_spans = self.apply_anno_fixes(doc_name, text, etype, anno_spans)
                spans, mention = zip(*anno_spans)
                mention = "...".join(mention)
                annotations.append(Annotation(doc_name, spans, etype, mention))

            except Exception as e:
                print(f"Annotation parsing error: {doc_name} {item}")

        return annotations

    def load_annotations(self, filelist):
        annotations = {}
        for doc_fpath, anno_fpath in filelist:
            doc_name = doc_fpath.split('/')[-1]
            annotations[doc_name] = self._parse(doc_fpath, anno_fpath)
        return annotations
