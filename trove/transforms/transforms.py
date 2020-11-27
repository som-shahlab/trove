import re

class SmartLowercase:
    """
    Smarter text lowercasing
    """
    not_acronyms = {'HIP', 'LEG', 'SKIN', 'LUNG', 'RIB', 'ARM',
                    'BACK', 'NECK', 'DUNG', 'CONJ', 'EYE', 'HEAD',
                    'HEEL', 'FOOT', 'TOE', 'FACE', 'NOSE', 'WOMB', 'LIMB',
                    'JAW', 'BONE', 'CELL', 'HOCK', 'PALM', 'MID', 'KNEE',
                    'ANAL', 'ANUS', 'VEIN', 'NAIL', 'SHIN', 'DUCT', 'ORAL',
                    'THE', 'EYES', 'LEFT', 'RIGHT'}

    def __call__(self, term):

        if term in SmartLowercase.not_acronyms:
            return term.lower()

        if term[0].isupper() and term[1:].islower():
            return term.lower() if len(term) > 2 else term

        # Check whether something is an upper-cased acronym.
        # We assume acronyms are less than 5 letters long.
        if term.isupper() and len(term) < 5:
            return term

        # eNOS | mDNA | GnBR
        if re.search(r'''^([A-Z][a-z][A-Z]{1,}|[a-z][A-Z]{3,})$''', term):
            return term

        # Handle the special case of vertebrae, which have the form
        # [Letter][Number] e.g. T9, wherein the phrase should be
        # capitalized on the first letter.
        # TODO: Handle the case of the hyphen in between letter/number
        if re.search('^[a-zA-Z][0-9]+$', term):
            return term.upper().strip()

        # Otherwise just lowercase everything
        return term.lower()