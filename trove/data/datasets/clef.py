"""
Load the ShAReCLEF 2014 Task 2 corpus and convert to BRAT objects

"""
import re
import glob
import json
import pdb
from collections import namedtuple
from ..dataloaders.contexts import Span
from ..dataloaders.brat import Entity, Relation
from trove.labelers.taggers import Tagger
from trove.labelers.taggers.timex import TimexNormalizer


fields = [
    'disorder',
    'negation',
    'subject',
    'uncertainty',
    'course',
    'severity',
    'conditional',
    'generic',
    'bodylocation',
    'timex',
    'doctime'
]
CLEFSlotFilled = namedtuple('CLEFSlotFilled', fields)

def get_spans(s):
    return tuple([tuple(map(int, i.split("-"))) for i in s.split(",")])

def get_cue(s):
    return tuple(get_spans(s)) if s != 'null' else None

def get_text(spans, document):
    if not spans:
        return
    m = []
    for s in spans:
        i,j = s
        m += [document[i:j]]
    return "...".join(m)

def get_entity(class_norm, class_name, cue, doc_name, document):
    """
    Initalize Entity object
    """
    if not cue or cue in ['null', 'nul', 'unmarked']:
        return None
    eid = (doc_name, cue)
    cue = get_cue(cue)
    m = Entity(eid, doc_name, class_name, cue, get_text(cue, document))
    if class_norm:
        m.attribs[class_norm.strip()] = 1
    return m


def get_span_sentence(span, document):
    start = min([i for offsets in span for i in offsets])
    for sent in document.sentences:
        end = len(sent.words[-1])
        if start >= sent.abs_char_offsets[0] and \
                start <= (sent.abs_char_offsets[-1] + end):
            return sent
    return None


class Clef2014Dataset(object):

    def __init__(self, anno_dir, doc_dir, splits_filepath):
        self.anno_dir = anno_dir
        self.doc_dir  = doc_dir
        self.splits_filepath = splits_filepath
        self.types = [
            'Negation',
            'Subject',
            'Uncertainty',
            'Course',
            'Severity',
            'Conditional',
            'Generic',
            'BodyLocation',
            'Timex'
        ]
        self.annotations = self.get_annotations()
    
    def get_annotations(self):
        
        anno_map, doc_map = {}, {}
        anno_splits = {'train':[], 'dev':[], 'test':[]}
        
        # Read in the splits from the file such that splits['train']
        # corresponds to a list with all the document names associated
        # with the training set
        with open(self.splits_filepath, 'r') as f:
            splits = json.load(f)
        
        # Turn splits[k] into a set for fast lookup
        splits = {k: set(v) for k, v in splits.items()}
        
        # Build a dictionary that maps document names to
        # the split in which they belong
        doc_to_split = {doc_name:split 
                        for split, doc_list in splits.items() 
                        for doc_name in doc_list}
        
        # annotations
        anno_filelist = glob.glob(f'{self.anno_dir}/*')
        for fname in anno_filelist:
            doc_name = fname.split("/")[-1].split(".")[0]
            anno_map[doc_name] = fname
            anno_splits[doc_to_split[doc_name]].append(doc_name)
        
        # documents
        docs_filelist = glob.glob(f'{self.doc_dir}/*')
        for fname in docs_filelist:
            doc_name = fname.split("/")[-1].split(".")[0]
            doc_map[doc_name] = fname
        
        self.splits = splits
        self.anno_splits = anno_splits
        return self._load(doc_map, anno_map)
        # annos = glob.glob(f"{self.anno_dir}/*")
        # docs  = glob.glob(f"{self.doc_dir}/*")
        # return self._load(docs, annos)

    def _fix_annotation_errs(self, rela):
        """
        There are a few annotation errors involving span offsets,
        e.g., including a trailing/leading whitespace or skipping a trailing/leading
        character. This function fixes those cases in the gold dataset.

        :param rela:
        :return:
        """
        disorder = rela['disorder']

        # " Paradoxic septal motion" -> "Paradoxic septal motion"
        if disorder.doc_name == '18908-109838-ECHO_REPORT' and disorder.span[0] == (886, 910):
            entity = Entity(disorder.id_,
                            disorder.doc_name,
                            disorder.type_,
                            ((887, 910),),
                            'Paradoxic septal motion')
            entity.attribs = {key:value for key,value in disorder.attribs.items()}
            rela['disorder'] = entity

        # "pain medication" -> "pain medications"
        elif disorder.doc_name == '04303-005081-DISCHARGE_SUMMARY' and disorder.span[0] ==  (5854, 5869):
            entity = Entity(disorder.id_,
                            disorder.doc_name,
                            disorder.type_,
                            ((5854, 5870),),
                            'pain medications')
            entity.attribs = {key: value for key, value in disorder.attribs.items()}
            rela['disorder'] = entity

        # " paraseptal emphysema" -> "paraseptal emphysema"
        elif disorder.doc_name == '13748-001753-DISCHARGE_SUMMARY' and disorder.span[0] ==  (5560, 5581):
            entity = Entity(disorder.id_,
                            disorder.doc_name,
                            disorder.type_,
                            ((5561, 5581),),
                            'paraseptal emphysema')
            entity.attribs = {key: value for key, value in disorder.attribs.items()}
            rela['disorder'] = entity

        # " chronic obstructive pulmonary disease" -> "chronic obstructive pulmonary disease"
        elif disorder.doc_name == '22567-017288-DISCHARGE_SUMMARY' and disorder.span[0] ==  (8876, 8914):
            entity = Entity(disorder.id_,
                            disorder.doc_name,
                            disorder.type_,
                            ((8877, 8914),),
                            'chronic obstructive pulmonary disease')
            entity.attribs = {key: value for key, value in disorder.attribs.items()}
            rela['disorder'] = entity


    def _load_doc(self, fpath, document):
        """
        """
        import collections
        debug = collections.defaultdict(int)

        relations = []
        with open(fpath,"r") as fp:
            for line in fp:
                row = re.split("[|]+", line.strip())
                doc_name = row[0].split(".")[0]
                rela = {}

                cui = row[17]
                if cui.lower() == 'null':
                    debug['NULL'] += 1
                elif cui.strip()[0] == 'C':
                    debug['CUI'] += 1
                else:
                    debug[cui] += 1
                
                # disorder
                cue, class_norm = row[1:3]
                rela['disorder'] = get_entity(class_norm, "Disorder", cue, doc_name, document)
                
                # disorder attributes
                for i, class_name in zip(list(range(3,19,2)) + list(range(20,22,2)), self.types):
                    class_norm, cue = row[i:i+2]
                    m = get_entity(class_norm, class_name, cue, doc_name, document)
                    rela[class_name.lower()] = m
                
                # temporal relation between a disorder and document authoring time
                rela['doctime'] = row[19]

                # apply manual fixes for dataset errors
                self._fix_annotation_errs(rela)

                relations.append(CLEFSlotFilled(**rela))
        if len(debug) > 2:
            print(fpath, "\n", debug)
        return relations
        
    def _load(self, doc_map, anno_map):
        annotations = {}
        for doc_name in anno_map:
            anno_filepath = anno_map[doc_name]
            doc_text = open(doc_map[doc_name], 'r').read()
            annotations[doc_name] = self._load_doc(anno_filepath, doc_text)
        
        return annotations


class CLEFLabelsTagger(Tagger):
    """
    Use manual labels from the 2014 ShARe/CLEF Task2 dataset to generate spans.
    """
    def __init__(self, anno_dir, doc_dir, splits_filepath, target='disorder', multi_span_rule='ignore'):
        self.data = Clef2014Dataset(anno_dir=anno_dir, doc_dir=doc_dir, splits_filepath=splits_filepath)
        self.annotations = self.data.get_annotations()
        self.target = target
        self.multi_span_rule = multi_span_rule

    def tag(self, document, ngrams=None):
        """
        Use manually labeled data to generate Span objects
        """

        try:
            doc_annos = self.annotations[document.name]
        except:
            print(document.name, ' Not found on annotations')
            return None
        entities = {sent.i: {} for sent in document.sentences}

        for anno in doc_annos:
            entity = anno._asdict()[self.target]
            if entity is None:
                continue

            # skip multi-spans
            if self.multi_span_rule == 'ignore' and len(entity.span) > 1:
                continue

            sent = get_span_sentence(entity.span, document, entity.text)

            # sent_range = range(sent.abs_char_offsets[0], sent.abs_char_offsets[-1] + 1)
            # flag = ''
            # for s in entity.span:
            #     i,j = s
            #     if i not in sent_range:
            #         flag = '**'
            # print(entity, entity.span, sent_range[0], sent_range[-1], flag)

            if sent is None:
                continue

            # for multi-span entities, choose an anchor span (head or tail)
            if len(entity.span) == 1 or self.multi_span_rule == 'head':
                i, j = entity.span[0]
            else:
                i, j = entity.span[-1]

            offset = sent.abs_char_offsets[0]
            span = Span(i-offset, j-1-offset, sentence=sent)
            if self.target.upper() not in entities[sent.i]:
                entities[sent.i][self.target.upper()] = []
            entities[sent.i][self.target.upper()].append(span)
        
        document.annotations.update(entities)


def get_span_sentence(span, document, text=None):
    start = min([i for offsets in span for i in offsets])
    for sent in document.sentences:
        end = 0 #len(sent.words[-1])
        if start >= sent.abs_char_offsets[0] and start <= (sent.abs_char_offsets[-1] + end):
            return sent
    return None


def anno_to_spans(annotations, documents, target='disorder', multi_span_rule='ignore'):
    """
    Convert BRAT CLEF annotations to Span objects
    """
    spans = []
    timex_norm = TimexNormalizer()

    classes = {
        'doctimes' : {'OVERLAP':0, 'BEFORE_OVERLAPS':1, 'BEFORE':2, 'AFTER':3, 'UNK':4},
        'severity' : {'slight': 0, 'moderate': 1, 'severe': 2, 'unmarked': 3}
    }

    doc_idx = {doc.name:doc for doc in documents}
    for anno in annotations:
        entity = anno._asdict()[target]
        if entity is None:
            continue
        # skip annotations that don't have mtching docs
        if entity.doc_name not in doc_idx:
            continue

        # skip multi-spans
        if multi_span_rule == 'ignore' and len(entity.span) > 1:
            continue

        doc = doc_idx[entity.doc_name]


        #if doc.name == '10644-007491-DISCHARGE_SUMMARY':
        #    print(entity, entity.span)

        # for multi-span entities, choose an anchor span (head or tail)
        if len(entity.span) == 1 or multi_span_rule == 'head':
            i, j = entity.span[0]
        else:
            i, j = entity.span[-1]

        # use anchor span
        sent = get_span_sentence(((i,j),), doc)
        if sent is None:
            print("NO SENTENCE", entity)
            continue


        doctime  = classes['doctimes']['UNK'] if anno.doctime not in classes['doctimes'] else classes['doctimes'][anno.doctime]

        severity = classes['severity']['unmarked']
        if anno.severity:
            label = list(anno.severity.attribs.keys())[0]
            severity = classes['severity'][label]

        offset = sent.abs_char_offsets[0]
        span = Span(i-offset, j-1-offset, sentence=sent)
        span.props['negation'] = 1 if anno.negation else 0
        span.props['doctime']  = doctime
        span.props['severity'] = severity
        span.props['bodylocation_span'] = None
        span.props['bodylocation'] = None
        if anno.bodylocation:
            start, end = anno.bodylocation.span[0]
            span.props['bodylocation_span'] = Span(start-offset, end-1-offset, sentence=sent)
            span.props['bodylocation'] = list(anno.bodylocation.attribs.keys())[0]

        # convert subject to boolean (patient or not patient)
        span.props['subject'] = 0
        if anno.subject:
            span.props['subject'] = 1

        span.props['uncertainty'] = 0
        if anno.uncertainty:
            span.props['uncertainty'] = 1

        if anno.timex:
            ts = timex_norm._normalize_timex_str(anno.timex.text)
            span.props['timex'] = ts.date() if ts else anno.timex.text
        else:
            span.props['timex'] = None

        spans.append(span)

    return spans
