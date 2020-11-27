import re
import string


###############################################################################
#
# Term Transforms
#
###############################################################################

NOT_ACRONYMS = {'HIP', 'LEG', 'SKIN', 'LUNG', 'RIB', 'ARM',
                'BACK', 'NECK', 'DUNG', 'CONJ', 'EYE', 'HEAD',
                'HEEL', 'FOOT', 'TOE', 'FACE', 'NOSE', 'WOMB', 'LIMB',
                'JAW', 'BONE', 'CELL', 'HOCK', 'PALM', 'MID', 'KNEE',
                'ANAL', 'ANUS', 'VEIN', 'NAIL', 'SHIN', 'DUCT', 'ORAL',
                'THE', 'EYES', 'LEFT', 'RIGHT'}


def lowercase(t:str) -> str:
    """Smarter text normalization/lower casing"""
    if t in NOT_ACRONYMS:
        return t.lower()

    if t[0].isupper() and t[1:].islower():
        return t.lower() if len(t) > 2 else t

    # Check whether something is an upper-cased acronym
    # (We assume acronyms are less than 5 letters long)
    if t.isupper() and len(t) < 5:
        return t

    # eNOS | mDNA | GnBR
    if re.search(r'''^([A-Z][a-z][A-Z]{1,}|[a-z][A-Z]{3,})$''', t):
        return t

    # Handle the special case of vertebrae, which have the form
    # [Letter][Number] e.g. T9, wherein the phrase should be
    # capitalized on the first letter.
    # TODO: Handle the case of the hyphen in between letter/number
    if re.search('^[a-zA-Z][0-9]+$', t):
        return t.upper().strip()

    # Otherwise just lowercase everything
    return t.lower()


bracket_terms = ['combined oral contraceptive', 'gastro-intestinal use',
                 'appetite suppressants', 'erectile dysfunction',
                 'musculoskeletal use', 'gynaecological use',
                 'cardiovascular use', 'gynecological use',
                 'generic additions', 'antihypertensive', 'epilepsy control',
                 'respiratory use', 'no preparations', 'disease/finding',
                 'antidepressant', 'antiarrhythmic', 'anti-rheumatic',
                 'endocrine use', 'no drugs here', 'antihistamine',
                 'antispasmodic', 'oropharyngeal', 'antipsychotic',
                 'cns narcotic', 'anaesthesia', 'anesthesia', 'malignancy',
                 'parkinsons', 'anxiolytic', 'ambiguous', 'endocrine',
                 'analgesic', 'b-blocker', 'systemic', 'hypnotic', 'antacid',
                 'nausea', 'asthma', 'e,n,x', 'mouth', 'cough', 'skin', 'nose',
                 '123i', 'eyes', 'acne', '131i', 'obs', 'eye', 'ear', '1,5',
                 'z42', 'dup', 'z6', 'd', '2', '1']
parans_terms = ['context\\-dependent category', 'morphologic abnormality',
                'navigational concept', 'unit of presentation',
                'geographic location', 'religion/philosophy',
                'observable entity', 'assessment scale', 'qualifier value',
                'physical object', 'record artifact', 'body structure',
                'regime/therapy', 'cell structure', 'ethnic group',
                'environment', 'isbt symbol', 'disposition', 'occupation',
                'substance', 'procedure', 'attribute', 'situation',
                'geriatric', 'disorder', 'organism', 'function', 'specimen',
                'property', 'clinical', 'product', 'finding', '& level',
                'general', 'diffuse', 'person', 'event', 'cell']


def strip_affixes(t):
    suffixes = [
        'morphologic abnormality',
        'finding',
        'situation',
        'substance',
        'product',
        'navigational concept',
        'observable entity',
        'qualifier value',
        'ambiguous',
        'function',
        'category',
        'disorder',
        'clinical',
        '[&] congenital',
        'context\-dependent category',
        'event',
        'dup',
        'obs'
    ]
    suffixes += ['aromatic', 'in water', 'acne', 'parkinsons', 'oropharyngeal',
                 'procedure',
                 'endocrine use', 'nose', 'eye', 'ear', 'skin']

    suffixes = parans_terms + bracket_terms

    suffixes = sorted(suffixes, key=len, reverse=1)
    suffixes = f"[(\[]({'|'.join(suffixes)})[)\]]"
    rgx = f'(^\[[xdmq]\])|(([,]\s*(nos|unspecified|[-]retired[-])|{suffixes})$)'
    m = re.search(rgx, t, re.I)

    while m:
        t = t.replace(m.group(), "").strip()
        m = re.search(rgx, t, re.I)
    return t


class MetaNorm(object):
    """
    Normalize UMLS Metathesaurus concept strings.
    """
    def __init__(self, function=lambda x: x):
        # TTY in [OF,FN] suffixes
        suffixes = ['qualifier value', 'life style', 'cell structure',
                    'domestic', 'bird', 'organism',
                    'context\\-dependent category', 'inactive concept',
                    'navigational concept', 'lck', 'record artifact',
                    'core metadata concept', 'substance', 'event',
                    'organism', 'person', 'attribute', 'procedure',
                    'tumor staging', 'a', 'cell', 'chloroaniline',
                    'product', 'specimen', 'observable entity',
                    'racial group', 'si', 'namespace concept',
                    'environment', 'social concept', 'ras', 'unspecified',
                    'special concept', 'staging scale', 'disorder',
                    'geographic location', 'occupation', 'ethnic group',
                    'body structure', 'situation', 'physical force',
                    'trans', 'finding', 'epoxymethano', 'linkage concept',
                    'assessment scale', 'metadata', 'link assertion',
                    'dithiocarbamates', 'foundation metadata concept',
                    'morphologic abnormality', 'physical object']
        # , (PHRASE)
        strip_phrases = [
            'with other specified complications',
            'not elsewhere classified',
            'other specified sites',
            'other',
            'unspecified',
            'specified',
            'other specified',
            'site unspecified',
            'unspecified site',
            'unspecified type',
            'and unspecified',
            'episode of care unspecified',
            'nos',
            'os'
        ]
        self.strip_phrase_rgx = "[,] (?!:(with(out)*|and|in)*)({})$".format(
            "|".join(sorted(strip_phrases, key=len, reverse=1))
        )
        self.of_fn_rgx = "\(({})\)$".format("|".join(
            sorted(suffixes, key=len, reverse=1))
        )
        self.function = function

    def normalize(self, s):
        '''
        Heuristics for stripping non-essential UMLS string clutter

        :param s:
        :return:
        '''

        s = re.sub(r'''^\[[xd]\]''', "", s)

        s = s.replace("--", " ")
        s = re.sub("[(\[<].+[>)\]]$", "", s)
        s = re.sub("(\[brand name\]|[,]* NOS)+", "", s).strip()
        s = s.strip().strip("_").strip(":")
        s = re.sub("(\[.{1}\])+", "", s).strip()
        s = re.sub("\-RETIRED\-$", "", s, re.I).strip()
        s = re.sub("BOLD[:].+$", "", s).strip()
        s = re.sub(" @ @ ", " ", s).strip()

        # remove phrases of the form "XXX, uspecified"
        s = re.sub(self.strip_phrase_rgx, "", s).strip()
        s = re.sub(r'''^(other and ((un)*specified|ill-defined)''' \
                   '''|specified|other|with|and|in) ''', "", s).strip()
        s = re.sub(r''' (nos|os)$''', "", s).strip()

        # normalize TTY in [OF,FN]
        s = re.sub(self.of_fn_rgx, "", s).strip()
        # remove digits/stray punctuation
        s = re.sub(
            "^([0-9]+[{}]*)+$".format(string.punctuation), "", s
        ).strip()

        # unicode for |
        s = s.split("&#x7c;")[0]

        # custom normalize function
        s = self.function(s)

        return s
