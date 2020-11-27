import re
from trove.contrib.labelers.clinical.taggers.negex import NegEx
from trove.contrib.labelers.clinical.taggers import (
    match_regex, token_distance, Ngrams, dict_matcher,
    get_left_span, get_right_span
)

ABSTAIN = 0
EXPOSURE = 1
NO_EXPOSURE = 2

negex = NegEx(f'../data/supervision/negex')


#
# Helper Functions
#

def is_negated(span, window=None):
    left = get_left_span(span, window=window)
    trigger = match_regex(negex.rgxs['definite']['left'], left)
    return True if trigger else False


#
# Labeling Functions
#

def LF_covid_contact(s):
    rgx = r'''\b(known|confirmed)*(\s)*(coronavirus|\+(\s)*covid|covid|covid\s19|covid-19)(\s)*(\+|positive|pos)(\s)*(contact(s)*|person(s)*|pt(s)*|patient(s)*)*\b'''
    trigger = match_regex(rgx, s)
    if not trigger:
        return ABSTAIN
    return EXPOSURE if not is_negated(trigger) else NO_EXPOSURE


def LF_contact_covid(s):
    rgx = r'''\b(known\s)*(contact(s)*)(\s)*(with|w\/)*(\s)*(known|confirmed)*(\s)*(coronavirus|covid|covid\s19|covid-19|covid(\s)*\+)(\scontact)*\b'''
    trigger = match_regex(rgx, s)
    if not trigger:
        return ABSTAIN
    return EXPOSURE if not is_negated(trigger) else NO_EXPOSURE


def LF_coworker(s):
    rgx = r'''\b(co-worker|coworker|co\sworker)\b'''
    v = re.search(rgx, s.text, re.I) is not None
    return EXPOSURE if v else ABSTAIN


def LF_sick_contacts(s):
    rgx = r'''\b(other|known)*(sick\scontact)\b'''
    trigger = match_regex(rgx, s)
    if not trigger:
        return ABSTAIN
    return EXPOSURE if not is_negated(trigger) else NO_EXPOSURE


def LF_family_covid_positive(s):
    rgx = r'''\b(coronavirus|\+covid|covid-19|covid|covid\s19)(\s)*(positive|\+)*\b'''
    trigger = match_regex(rgx, s)

    v = re.search(rgx, s.text, re.I) is not None
    rgx = r'''(family\smember(s)*|spouse|partner|husband|wife|son(s)?|daughter(s)?|child(ren)?|father|mother|mom|dad|parent(s)*|brother|sister|aunt|uncle|cousin|grandpa|grandma|grandparent(s)*)'''

    v &= re.search(rgx, s.text, re.I) is not None
    return EXPOSURE if (v and not is_negated(trigger)) else ABSTAIN


def LF_family_positive_covid(s):
    rgx = r'''\b(positive\sfor)(\s)*(coronavirus|covid-19|covid|covid\s19)\b'''
    v = re.search(rgx, s.text, re.I) is not None

    rgx = r'''(family\smember(s)*|spouse|partner|husband|wife|son(s)?|daughter(s)?|child(ren)?|father|mother|mom|dad|parent(s)*|brother|sister|aunt|uncle|cousin|grandpa|grandma|grandparent(s)*)'''
    v &= re.search(rgx, s.text, re.I) is not None

    return EXPOSURE if v else ABSTAIN


def LF_exposed_to(s):
    rgx = r'''\b(exposed\sto)\b'''
    trigger = match_regex(rgx, s)
    if not trigger:
        return ABSTAIN
    return EXPOSURE if not is_negated(trigger) else NO_EXPOSURE


def LF_exposure(s):
    rgx = r'''\b(exposure\sto\s)*(known)*(\s)*(coronavirus|\+covid|covid-19|covid|covid\s19)*(\s)*(positive|\+)*(\s)*(exposure)\b'''
    trigger = match_regex(rgx, s)
    if not trigger:
        return ABSTAIN
    return EXPOSURE if not is_negated(trigger) else NO_EXPOSURE


def LF_covid_ro(s):
    rgx = r'''\b(r\/o\scovid|covid\sr\/o|covid\srule\sout|(rule(\s)*(him|her|them)*\sout(\sfor)*\scovid))\b'''
    trigger = match_regex(rgx, s)
    return NO_EXPOSURE if not trigger else ABSTAIN


def LF_no_mention_covid(s):
    rgx = r'''\b(covid)(\+)?(-19\b)?'''
    trigger = match_regex(rgx, s)
    return NO_EXPOSURE if (not trigger) else ABSTAIN


def LF_confirmed_contacts(s):
    rgx = r'''\b(positive|confirmed)\scontact(s)*'''
    trigger = match_regex(rgx, s)
    if not trigger:
        return ABSTAIN
    return EXPOSURE if not is_negated(trigger) else ABSTAIN


def LF_exposure_2(s):
    rgx = r'''((\b(exposure)\b)|(\b(exposed\sto)\b))'''
    trigger = match_regex(rgx, s)
    if not trigger:
        return ABSTAIN
    return EXPOSURE if not is_negated(trigger) else NO_EXPOSURE


def LF_exposure_3(s):
    rgx = r'''(known)(\s)*(coronavirus|\+covid|covid-19|covid|covid\s19)(\s)*(exposure)'''
    trigger = match_regex(rgx, s)
    if not trigger:
        return ABSTAIN
    return EXPOSURE if not is_negated(trigger) else NO_EXPOSURE


def LF_exposure_4(s):
    rgx = r'''(exposure\sto\s)(coronavirus|\+covid|covid-19|covid|covid\s19)'''
    trigger = match_regex(rgx, s)
    if not trigger:
        return ABSTAIN
    return EXPOSURE if not is_negated(trigger) else NO_EXPOSURE


def LF_exposure_5(s):
    rgx = r'''(coronavirus|\+covid|covid-19|covid|covid\s19)(\s)*(exposure)'''
    trigger = match_regex(rgx, s)
    if not trigger:
        return ABSTAIN
    elif not is_negated(trigger):
        left_span = get_left_span(trigger, window=None)
        rgx = r'''(may|unknown|unsure)'''
        trigger = match_regex(rgx, left_span)
        if not trigger:
            return EXPOSURE
        else:
            return NO_EXPOSURE
    else:
        return NO_EXPOSURE


def LF_covid_negative(s):
    """
    COVID negative
    """
    rgx = r'''(coronavirus|\+covid|covid-19|covid|covid\s19)(\s)*(neg(ative)?)'''
    trigger = match_regex(rgx, s)
    if not trigger:
        return ABSTAIN
    else:
        return NO_EXPOSURE


def LF_covid_testing(s):
    """
    I am testing for
    """
    rgx = r'''test(ing|ed)(\s)*(for)*(\s)*(coronavirus|\+covid|covid-19|covid|covid\s19)'''
    trigger = match_regex(rgx, s)
    if not trigger:
        return ABSTAIN
    else:
        return NO_EXPOSURE


def LF_covid_biolerplate(s):
    """
    covid provider note
    Suspected Covid-19 Virus Infection
    """

    rgx = r'''((coronavirus|\+covid|covid-19|covid|covid\s19)(\s)*(provider note)|(susp(ected|icion) covid(-19)? virus (infection)?))'''
    trigger = match_regex(rgx, s)
    if not trigger:
        return ABSTAIN
    else:
        return NO_EXPOSURE


def LF_possible_contact(s):
    """
    possible contact with COVID
    """
    rgx = r'''possi(ble|bility)(\s)*(contact)*(\s)*(with)*(\s)*(coronavirus|\+covid|covid-19|covid|covid\s19)'''
    trigger = match_regex(rgx, s)
    if not trigger:
        return ABSTAIN
    else:
        return NO_EXPOSURE


lfs = [
    LF_exposure_2,
    LF_exposure_3,
    LF_exposure_4,
    LF_exposure_5,
    LF_covid_negative,
    LF_covid_testing,
    LF_covid_biolerplate,
    LF_possible_contact,
    LF_covid_contact,
    LF_sick_contacts,
    LF_family_covid_positive,
    LF_family_positive_covid,
    LF_coworker,
    LF_no_mention_covid
]
