from trove.contrib.labelers.clinical.taggers import Tagger
from trove.contrib.labelers.clinical.helpers import token_distance, overlaps


###############################################################################
#
# Time Delta Tagger
#
###############################################################################

class TimeDeltaTagger(Tagger):
    """
    Link spans to their nearest (in token distance) normalized date mention.
    For example:

    INPUTS: DOCTIME:  12/15/2017
            SPAN:     hip implant
            SENTENCE: On 11/13/2017, the patient came into the clinic with an
                      infected hip implant.
    OUTPUT: (hip implant, -32 days)

    CAVEATS:
    - We restrict to datetime mentions found in the same sentence as our
      target span.
    - We ignore spans that occur within HEADER mentions
    - Relying on only the nearest DATETIME mention fails in many cases.
      Consider the examples:

        + Chief Complaint: 60 year old white female s/p CABG 2016-01-05 with
          fever and sternal wound drainage.

      Coronary artery bypass grafting (CABG) was done on 2016-01-05,
      but 'fever' and 'sternal wound drainage' are current conditions.

        + Since the previous tracing of 2015-03-17 sinus bradycardia
          rate is slower.

      Current slower sinus bradycardia is being compared to a previous tracing
      on 2015-03-17.

    TODOs:
    - This should be a proper classification task where we predict links
     between spans and TIMEX3's.
    - Create interface for ranked list (by token distance) of all datetimes
      given a span to allow for more flexible LF design.

    """

    def __init__(self, targets):
        self.targets = targets

    def _apply_lfs(self, span):
        # TODO: Implement labeling functions for this task.
        pass

    def tag(self, document, **kwargs):
        """

        Parameters
        ----------
        document
        kwargs

        Returns
        -------

        """
        doc_ts = document.props['doctime']
        if not doc_ts:
            return

        for i, sent in enumerate(document.sentences):

            if 'TIMEX3' not in document.annotations[i]:
                continue

            # get sentence datetime mentions
            sent_ts = [ts for ts in document.annotations[i]['TIMEX3'] if
                       ts.normalized]
            if not sent_ts:
                continue

            hdr = document.annotations[i]['HEADER']

            for name in set(document.annotations[i]).intersection(
                    self.targets):
                for span in document.annotations[i][name]:
                    # ignore spans that overlap HEADER entities
                    if hdr[0] and overlaps(span, hdr[0]):
                        continue
                    # select closest DATETIME mention
                    dists = [token_distance(span, ts) for ts in sent_ts]
                    dists = sorted(zip(dists, sent_ts), key=lambda x: x[0],
                                   reverse=0)
                    tdelta = dists[0][-1]
                    span.props['tdelta'] = (tdelta.normalized - doc_ts).days
                    span.props['timex'] = tdelta.normalized
                    span.props['timex_span'] = tdelta