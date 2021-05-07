import re
import pandas as pd

from trove.contrib.labelers.clinical.helpers import get_left_span, get_right_span, token_distance
from trove.contrib.labelers.clinical.timex import rgx_number_full


###############################################################################
#
# Temporality THYME corpus 2014
#
###############################################################################


class TemporalityLabelingFunctions(object):

    def __init__(self, data_root):
        self.data_root = data_root
        self.class_map = self.load_class_map()
        self.term_class_map = self.load_term_class_map()
        self.load_rgx()

    def load_class_map(self):
        """

        """

        class_map = {
            'ABSTAIN': -1,
            'OVERLAPS': 0,
            'BEFORE_OVERLAPS': 1,
            'BEFORE': 2,
            'AFTER': 3,
        }

        return class_map

    def _get_rgx(self, file_lines: list, name: str):
        """
        Auxiliary function to read a rgx from the TSV file by passing a name
        """
        rgx_list = []
        for line in file_lines:
            label, name_line, tier, ignore_case, regex, notes = line.split(
                '\t')
            if name_line == name:
                rgx_list.append(regex)

        return rgx_list

    def load_term_class_map(self):
        """
        This loads a special file created with dictionary of words that are dominant in certain classes
        """
        term_class_map = open(
            f'{self.data_root}/doc_time_rel_dominant_class.tsv', 'r').read().splitlines()
        term_class_map = {x.split('\t')[0]: int(
            x.split('\t')[1]) for x in term_class_map}

        return term_class_map

    def load_word_lists(self):
        """
        This method loads the word list in the TSV file
        """
        all_rgx = open(f"{self.data_root}/thyme_regexes.tsv", 'r').readlines()

        # get all the distinct names
        name_list = pd.read_csv(
            f"{self.data_root}/thyme_regexes.tsv", sep="\t").loc[:, "NAME"].unique().tolist()

        for name in name_list:
            if name.startswith("word"):
                rgx_list_set = set(self._get_rgx(all_rgx, name))

                # Create the object inside the class with the same name as in the TSV file
                setattr(self, name, rgx_list_set)

                print(f"{name} Words Loaded")

    def load_rgx(self):
        """
        This method loads all the regex on the TSV file in the appopiate object
        """
        # Read File
        all_rgx = open(f"{self.data_root}/thyme_regexes.tsv", 'r').readlines()

        # get all the distinct names
        name_list = pd.read_csv(
            f"{self.data_root}/thyme_regexes.tsv", sep="\t").loc[:, "NAME"].unique().tolist()

        for name in name_list:
            if name.startswith("regex"):
                rgx_list_string = self._get_rgx(all_rgx, name)
                rgx_list_compiled = [re.compile(
                    rgx, re.I) for rgx in rgx_list_string]

                # Create the object inside the class with the same name as in the TSV file
                setattr(self, name, rgx_list_compiled)

                print(f"{name} Regexes Loaded")

    def LF_overlap_event(self, span):
        evts = {'hemorrhage', 'edema', 'complications',
                'distress', 'ischemia', 'pneumothorax'}
        return self.class_map["OVERLAPS"] if span.text.lower() in evts else self.class_map["ABSTAIN"]

    def LF_regex_before_left(self, span):
        sentence_text = get_left_span(span, span.sentence, window=5).text
        for rgx in self.regex_before_left:
            if rgx.search(sentence_text):
                return self.class_map["BEFORE"]
            else:
                continue
        return self.class_map["ABSTAIN"]

    def LF_regex_before_overlap_left(self, span):
        left = get_left_span(span, span.sentence, window=4)
        context = f"{left.text} {span.text}"
        for rgx in self.regex_before_overlap_left:
            if rgx.search(context):
                return self.class_map["BEFORE_OVERLAPS"]
            else:
                continue
        return self.class_map["ABSTAIN"]

    def LF_regex_after_left(self, span):
        rgxs_stop_words = [re.compile(stop_word, re.I)
                           for stop_word in ['otherwise', 'today']]
        left = get_left_span(span, span.sentence, window=10)
        small_left = get_left_span(span, span.sentence, window=2)
        context = f"{left.text} {span.text}"
        # Filter stopwords
        for rgx_stop in rgxs_stop_words:
            if rgx_stop.search(small_left.text):
                return self.class_map["ABSTAIN"]
            else:
                continue

        for rgx in self.regex_after_left:
            if rgx.search(context) and span.text not in ['arranged',
                                                         'agreed',
                                                         'consented',
                                                         'discussed',
                                                         'plan',
                                                         'plans',
                                                         'scheduled']:
                return self.class_map["AFTER"]
            else:
                continue
        return self.class_map["ABSTAIN"]

    def LF_word_list_before(self, span):
        if span.text.lower() in self.word_list_before:
            return self.class_map["BEFORE"]
        else:
            return self.class_map["ABSTAIN"]

    def LF_word_list_overlap(self, span):
        if span.text.lower() in self.word_list_overlap:
            return self.class_map["OVERLAPS"]
        else:
            return self.class_map["ABSTAIN"]

    def LF_sections_overlap(self, span):
        if span.props['section'] is not None:
            if span.props['section'].text.lower() in self.sections_overlap:
                return self.class_map["OVERLAPS"]
            else:
                return self.class_map["ABSTAIN"]
        else:
            return self.class_map["ABSTAIN"]

    def LF_history_of(self, span):
        rgx = r'''((family|surgical|oncologic|medical|patient) history|history of|history:|(his|her) history)'''
        left = get_left_span(span, span.sentence, window=2)
        right = get_right_span(span, span.sentence, window=2)
        context = f"{left.text} {span.text} {right.text}"
        return self.class_map["BEFORE_OVERLAPS"] if re.search(rgx, context, re.I) else self.class_map["ABSTAIN"]

    def _is_hypothetical(span):
        accept_rgxs = [
            r"\b(if need be)\b",
            r"\b((if|should)\s+(you|she|he|be)|(she|he|you)\s+(might|could|may)\s*(be)*|if)\b",
            r"\b((possibility|potential|chance|need) (for|of)|potentially)\b",
            r"\b(candidate for|pending)\b",
            r"\b(assuming)\s+(you|she|he)\b",
            r"(recommendation)\s*[:]",
            r"(planned procedure)\s*[:]",
            r"\b(upcoming|would benefit from|(undergo|requires) a)\b",
            r'''\b(please call or return (for|if))\b''',
            r"\b(proceed with|consider|to undergo|scheduled for)\b"
        ]

        text = get_left_span(span, span.sentence, window=20).text

        for rgx in accept_rgxs:
            if re.search(rgx, text, re.I):
                return True

        return False

    def LF_hypothetical(self, span):
        return self.class_map["AFTER"] if self._is_hypothetical(span) else self.class_map["ABSTAIN"]

    def LF_tdelta_after_dist_1(self, span):
        """ requires revision date info """
        if 'tdelta' not in span.props or 'doctime' not in span.sentence.document.props:
            return self.class_map["ABSTAIN"]
        closest_ts = span.props['timex_span']
        tdelta_start = span.props['tdelta']
        tdelta_rev = closest_ts - span.sentence.document.props['doctime']

        v = tdelta_start > 5 and tdelta_rev > 5
        dist = token_distance(span, span.props['timex_span'])
        return self.class_map["OVERLAPS"] if v and dist <= 1 else self.class_map["ABSTAIN"]

    def LF_tdelta_overlaps_dist_1(self, span):
        if 'tdelta' not in span.props:
            return self.class_map["ABSTAIN"]
        v = span.props['tdelta'] >= 0
        dist = token_distance(span, span.props['timex_span'])
        return self.class_map["OVERLAPS"] if v and dist <= 1 else self.class_map["ABSTAIN"]

    def LF_tdelta_overlaps_dist_5(self, span):
        if 'tdelta' not in span.props:
            return self.class_map["ABSTAIN"]
        v = span.props['tdelta'] >= 0
        dist = token_distance(span, span.props['timex_span'])
        return self.class_map["OVERLAPS"] if v and (dist > 1 and dist <= 5) else self.class_map["ABSTAIN"]

    def LF_tdelta_overlaps_dist_10(self, span):
        if 'tdelta' not in span.props:
            return self.class_map["ABSTAIN"]
        v = span.props['tdelta'] >= 0
        dist = token_distance(span, span.props['timex_span'])
        return self.class_map["OVERLAPS"] if v and (dist > 5 and dist <= 10) else self.class_map["ABSTAIN"]

    def LF_tdelta_overlaps_dist_long(self, span):
        if 'tdelta' not in span.props:
            return self.class_map["ABSTAIN"]
        v = span.props['tdelta'] >= 0
        dist = token_distance(span, span.props['timex_span'])
        return self.class_map["OVERLAPS"] if v and dist > 10 else self.class_map["ABSTAIN"]

    def LF_tdelta_before_dist_1(self, span):
        if 'tdelta' not in span.props:
            return self.class_map["ABSTAIN"]
        v = span.props['tdelta'] < -1
        dist = token_distance(span, span.props['timex_span'])
        return self.class_map["BEFORE"] if v and dist <= 1 else self.class_map["ABSTAIN"]

    def LF_tdelta_before_dist_5(self, span):
        if 'tdelta' not in span.props:
            return self.class_map["ABSTAIN"]
        v = span.props['tdelta'] < -1
        dist = token_distance(span, span.props['timex_span'])
        return self.class_map["BEFORE"] if v and (dist > 1 and dist <= 5) else self.class_map["ABSTAIN"]

    def LF_tdelta_before_dist_10(self, span):
        if 'tdelta' not in span.props:
            return self.class_map["ABSTAIN"]
        v = span.props['tdelta'] < -1
        dist = token_distance(span, span.props['timex_span'])
        return self.class_map["BEFORE"] if v and (dist > 5 and dist <= 10) else self.class_map["ABSTAIN"]

    def LF_tdelta_before_dist_long(self, span):
        if 'tdelta' not in span.props:
            return self.class_map["ABSTAIN"]
        v = span.props['tdelta'] < -1
        dist = token_distance(span, span.props['timex_span'])
        return self.class_map["BEFORE"] if v and dist > 10 else self.class_map["ABSTAIN"]

    def LF_overlaps_now(self, span):
        left = get_left_span(span, span.sentence, window=1).text
        right = get_right_span(span, span.sentence, window=1).text
        rgx = r'''\b(now)\b'''
        return self.class_map["OVERLAPS"] if re.search(rgx, left, re.I) or re.search(rgx, right, re.I) else self.class_map["ABSTAIN"]

    def LF_overlaps_current(self, span):
        left = get_left_span(span, span.sentence, window=4)
        return self.class_map["OVERLAPS"] if re.search(r'''\b(current(ly)*)\b''', left.text, re.I) else self.class_map["ABSTAIN"]

    def LF_before_recent(self, span):
        left = get_left_span(span, span.sentence, window=2)
        return self.class_map["BEFORE"] if re.search(r'''\b(recent(ly)*)\b''', left.text, re.I) else self.class_map["ABSTAIN"]

    def LF_before_x_ago(self, span):
        rgx = r'''\b(([1-9][0-9]|{}|few|a) ((year|month|week|day|hour)[s]*) ago)\b'''.format(rgx_number_full)
        left = get_left_span(span, span.sentence, window=5).text
        right = get_right_span(span, span.sentence, window=5).text
        return self.class_map["BEFORE"] if re.search(rgx, left, re.I) or re.search(rgx, right, re.I) else self.class_map["ABSTAIN"]

    def LF_after_next_x(self, span):
        rgx = r'''\b((next|upcoming) (month|week|monday|tuesday|wednesday|thursday|friday)|(later (today|tonight|date|this (week|month|afternoon|evening))))\b'''.format(
            rgx_number_full)
        left = get_left_span(span, span.sentence, window=5).text
        right = get_right_span(span, span.sentence, window=5).text
        return self.class_map["AFTER"] if re.search(rgx, left, re.I) or re.search(rgx, right, re.I) else self.class_map["ABSTAIN"]

    def LF_after_tomorrow(self, span):
        left = get_left_span(span, span.sentence, window=5).text
        right = get_right_span(span, span.sentence, window=5).text
        rgx = r'''\b(tomorrow)\b'''
        return self.class_map["AFTER"] if re.search(rgx, left, re.I) or re.search(rgx, right, re.I) else self.class_map["ABSTAIN"]

    def LF_before_yesterday(self, span):
        left = get_left_span(span, span.sentence, window=2).text
        right = get_right_span(span, span.sentence, window=2).text
        rgx = r'''\b(yesterday)\b'''
        return self.class_map["BEFORE"] if re.search(rgx, left, re.I) or re.search(rgx, right, re.I) else self.class_map["ABSTAIN"]

    def LF_after_will(self, span):
        left = get_left_span(span, span.sentence, window=6).text
        rgx = r'''\b(will (try)*|plan(ning)*)\b'''
        return self.class_map["AFTER"] if re.search(rgx, left, re.I) else self.class_map["ABSTAIN"]

    def LF_after_should(self, span):
        left = get_left_span(span, span.sentence, window=10).text
        rgx = r'''\b(should (be)*)\b'''
        return self.class_map["AFTER"] if re.search(rgx, left, re.I) else self.class_map["ABSTAIN"]

    def LF_dominant_temporality_terms(self, span):
        t = span.text.lower()
        return self.class_map["ABSTAIN"] if t not in self.term_class_map else self.term_class_map[t] - 1

    def lfs(self):
        """

        Parameters
        ----------

        Returns
        -------

        """

        lfs = [
            self.LF_tdelta_overlaps_dist_1,
            self.LF_tdelta_overlaps_dist_5,
            self.LF_tdelta_overlaps_dist_10,
            self.LF_tdelta_overlaps_dist_long,
            self.LF_overlaps_now,
            self.LF_overlap_event,
            self.LF_overlaps_current,
            self.LF_word_list_overlap,
            self.LF_sections_overlap,
            self.LF_tdelta_before_dist_1,
            self.LF_tdelta_before_dist_5,
            self.LF_tdelta_before_dist_10,
            self.LF_tdelta_before_dist_long,
            self.LF_before_x_ago,
            self.LF_before_recent,
            self.LF_before_yesterday,
            self.LF_regex_before_left,
            self.LF_word_list_before,
            self.LF_history_of,
            self.LF_regex_before_overlap_left,
            self.LF_hypothetical,
            self.LF_after_next_x,
            self.LF_after_tomorrow,
            self.LF_after_will,
            self.LF_after_should,
            self.LF_regex_after_left,
            self.LF_dominant_temporality_terms
        ]

        print(f'Labeling Functions n={len(lfs)}')
        return lfs
