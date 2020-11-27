import re
import glob
import datetime
from lxml import etree
from ..dataloaders.contexts import Span
from ...labelers.taggers import Tagger
import pdb


#################################################################################
#
#  Dataset
#
#################################################################################

class ThymeDataset(object):

    def __init__(self, anno_dir, doc_dir):
        self.anno_dir = anno_dir
        self.doc_dir = doc_dir
        self.annotations = self.get_annotations()

    def get_annotations(self):

        anno_map, doc_map = {}, {}
        splits = {'train': [], 'dev': [], 'test': []}
        anno_splits = {'train': [], 'dev': [], 'test': []}

        for split in ['Train', 'Dev', 'Test']:
            # annotations
            filelist = glob.glob(f'{self.anno_dir}/{split}/*/*.Temporal-*.gold.completed.xml')
            for fname in filelist:
                doc_name = fname.split("/")[-1].split(".")[0]
                anno_map[doc_name] = fname
                anno_splits[split.lower()].append(doc_name)

            # documents
            filelist = glob.glob(f'{self.doc_dir}/{split.lower()}/*')
            for fname in filelist:
                doc_name = fname.split("/")[-1].split(".")[0]
                doc_map[doc_name] = fname
                # keep track of splits
                splits[split.lower()].append(doc_name)

        self.splits = splits
        self.anno_splits = anno_splits
        return self._load(doc_map, anno_map)

    def _load_annotations(self, fpath, text):
        """
        Load THYME XML
        TODO: Only loads entity annotations ATM
        """
        labels = {'EVENT': {}, 'TIMEX3': {}, 'TLINK': {}, 'SECTIONTIME': {}, 'DOCTIME': {}}
        root = etree.parse(fpath).getroot()
        annotations = root.xpath('//annotations/entity')

        for elem in annotations:
            try:
                _id = elem.xpath("id")[0].text
                span = elem.xpath("span")[0].text
                span = span.split(";")
                span = [list(map(int, s.split(","))) for s in span]
                etype = elem.xpath("type")[0].text

                if etype == 'EVENT':
                    # only events have temporal class
                    doc_time_rel = elem.xpath("properties/DocTimeRel")[0].text
                    polarity = elem.xpath("properties/Polarity")[0].text
                    labels[etype][_id] = (span, doc_time_rel, polarity)

                elif etype == 'TIMEX3':
                    eclass = elem.xpath("properties/Class")[0].text
                    i, j = span[0]
                    labels[etype][_id] = (span, eclass, text[i:j])

            except Exception as e:
                print(f'Error {e} {fpath}')

        return labels

    def _load(self, doc_map, anno_map):
        annotations = {}
        for doc_name in anno_map:
            text = open(doc_map[doc_name], 'r').read()
            annotations[doc_name] = self._load_annotations(anno_map[doc_name], text)
        return annotations


#################################################################################
#
#  Taggers
#
#################################################################################

class ThymeDocTimeTagger(Tagger):
    """
    THYME dataset docs includes markup for document creation & revistion times. 
    Use this as the document creation time.
    
    [meta rev_date="10/08/2010" start_date="10/07/2010" rev="0002"]
    
    """

    def __init__(self, doctime_field='start_date'):
        self.doctime_field = doctime_field

    def _parse_date_header(self, header, field):
        rgx = r'''{}="(.+?)[" ]'''
        m = re.search(rgx.format(field), header, re.I)
        return datetime.datetime.strptime(m.group(1), "%m/%d/%Y") if m else None

    def tag(self, doc):
        header = doc.text.split("\n")[0].strip()
        doc.props['start_date'] = self._parse_date_header(header, 'start_date')
        doc.props['rev_date'] = self._parse_date_header(header, 'rev_date')
        if doc.props['start_date'] is not None or doc.props['rev_date'] is not None:
            doc.props['doctime'] = doc.props['rev_date'] if self.doctime_field == 'rev_date' else doc.props[
                'start_date']
        else:
            print(f"No doctime found for {doc.name}")
            date_formats = ["%d%b%Y", "%m/%d/%Y", "%m/%d/%y", "%m-%d-%y", "%B %d%th, %Y", "%b-%d-%Y %H:%M:%S",
                            "%d-%b-%Y %H:%M:%S", "%d %b %Y", "%d %b %Y %I:%M%p", "%d%b%Y %I:%M%p", "%d-%b-%Y %I:%M",
                            "%d-%b-%Y", "%d-%b-%Y %H:%M", "%B %d, %Y"]
            doctime_candidates = []
            for key, value in doc.annotations.items():
                for key2, value2 in value.items():
                    if key2 == 'TIMEX3':
                        for date_span in value2:
                            for str_format in date_formats:
                                try:
                                    # doc.props['doctime'] = datetime.datetime.strptime(date_span.text, str_format)
                                    doctime_candidates.append(datetime.datetime.strptime(date_span.text, str_format))
                                except ValueError:
                                    continue
                            if len(doctime_candidates) == 0:
                                print(date_span.text)
            # get the maximum date since the earliest date could be date of birth
            if len(doctime_candidates) == 0:
                print(f"****NO DATE FOUND ON {doc.name} ****")
                doc_doctime = None
                # temporal fix to let the pipeline run to completion
                # doc.props['doctime'] = datetime.datetime.strptime("01/01/06 00:00", "%d/%m/%y %H:%M")
            else:
                # take the date difference between min and max
                # if the difference is greater than 1 year take the second date in order of sorting
                if len(doctime_candidates) > 1:
                    # get the unique ones
                    doctime_candidates = list(set(doctime_candidates))
                    # sort the candidates
                    doctime_candidates.sort()
                    date_diff = max(doctime_candidates) - min(doctime_candidates)
                    if date_diff.days > 365:
                        # get the second to avoid dates that could be birthdays or representing an event that within
                        # the time-frame of the note
                        doc_doctime = doctime_candidates[1]
                    else:
                        doc_doctime = doctime_candidates[0]
                else:
                    doc_doctime = doctime_candidates[0]

            doc.props['doctime'] = doc_doctime


class ThymeDocTimeTaggerV2(Tagger):
    """
    THYME dataset docs includes markup for document creation & revistion times.
    Use this as the document creation time.

    [meta rev_date="10/08/2010" start_date="10/07/2010" rev="0002"]

    """

    def __init__(self):
        pass

    def tag(self, document):
        header = document.text.split("\n")[0].strip()
        m = re.search(r'''start_date="(.+?)[" ]''', header, re.I)
        # if there is a match, extract the timestamp
        if m:
            print(m.group(1))
            ts = datetime.datetime.strptime(m.group(1), "%m/%d/%Y")
        else:
            ts = None
        document.props['doctime'] = ts


class ThymeLabelsTagger(Tagger):
    """
    Use manual labels from the TYHME dataset to generate spans. 
    """

    def __init__(self, anno_dir, doc_dir, target='EVENT'):

        self.data = ThymeDataset(anno_dir=anno_dir,
                                 doc_dir=doc_dir)
        self.annotations = self.data.get_annotations()
        self.target = target

    def get_span_sentence(self, span, document):
        start = min([i for offsets in span for i in offsets])
        for sent in document.sentences:
            end = len(sent.words[-1])
            if start >= sent.abs_char_offsets[0] and start <= (sent.abs_char_offsets[-1] + end):
                return sent
        return None

    def tag(self, document, **kwargs):
        """
        Use manually labeled data to generate Span objects
        """
        labels = self.annotations[document.name][self.target]
        doc_annos = [labels[_id] for _id in labels]
        entities = {sent.i: {} for sent in document.sentences}

        for anno in doc_annos:
            span, doc_time_rela, negated = anno
            sent = self.get_span_sentence(span, document)

            if sent is None:
                continue

            i, j = span[0]
            offset = sent.abs_char_offsets[0]
            span = Span(i - offset, j - 1 - offset, sentence=sent)

            if self.target.upper() not in entities[sent.i]:
                entities[sent.i][self.target.upper()] = []
            entities[sent.i][self.target.upper()].append(span)

        document.annotations.update(entities)
