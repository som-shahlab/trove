from datetime import datetime
from trove.contrib.labelers.clinical.helpers import *
from trove.contrib.labelers.clinical.taggers import Tagger


###############################################################################
#
# DocTime Tagger
#
###############################################################################

class DocTimeTagger(Tagger):

    def __init__(self, prop='doctime', format='%Y-%m-%d %H:%M:%S'):
        self.prop = prop
        self.format = format

    def tag(self, document, **kwargs):
        if self.prop not in document.props:
            document.props['doctime'] = None
        elif type(document.props[self.prop]) is str:
            ts = datetime.strptime(document.props[self.prop], self.format)
            document.props['doctime'] = ts

class TextFieldDocTimeTagger(Tagger):
    """
    Estimte document timestamp. Use either:
    1: Explicit note sign date of the form {field}:{datetime}, e.g.,
        T: 12-24-2005 11:30:00
    2: Most recent unambiguous TIMEX mention
    """

    def __init__(self,
                 targets=None,
                 field='T',
                 prop_name='doctime',
                 max_ts_default=True):

        self.targets = targets if targets else ['TIMEX3', 'HEADER']
        self.field = field
        self.prop_name = prop_name
        self.max_ts_default = max_ts_default

    def tag(self, document, **kwargs):

        max_date, sign_dates = None, []
        for i in range(len(document.annotations)):
            header = document.annotations[i]['HEADER'][0] \
                if 'HEADER' in document.annotations[i] else None
            timexes = document.annotations[i]['TIMEX3'] \
                if 'TIMEX3' in document.annotations[i] else []
            ts = [ts.normalized for ts in timexes if ts.normalized]

            if ts:
                max_date = max([max_date] + ts) if max_date else max(ts)

            if ts and header and re.search("^\s*{}[:]".format(self.field),
                                           header.text):
                sign_dates.extend(ts)

        if sign_dates:
            # defer to timestamps in the target field section
            document.props[self.prop_name] = max(sign_dates)
        elif max_date and self.max_ts_default:
            # select max from all date times
            document.props[self.prop_name] = max_date
        else:
            document.props[self.prop_name] = None


class MappedDocTimeTagger(Tagger):
    """
    Doctimes are provided in dictionary map of the form
        Dict[doc_name, datetime]
    """
    def __init__(self, doctimes):
        self.doctimes = doctimes

    def tag(self, document, **kwargs):
        document.props['doctime'] = self.doctimes[document.name] \
            if document.name in self.doctimes else None


