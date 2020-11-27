import re
import glob
import argparse
import numpy as np


def get_year(s):
    if not re.search(r'''[A-Za-z]+''', s):
        d = re.search(r'''([1-8][0-9]{3})[-][0-9]{1,2}[-][0-9]{1,2}''', s,
                      re.I)
        if d:
            return int(d.group(1))
    return None


def add_synthetic_dates(text, 
                        matches, 
                        min_date = 1900, 
                        max_date = 2020,
                        doc_ts = None,
                        preserve_offsets = True):
    """
    MIMIC data doesn't include real years in dates mentions. However,
    years are all offset by the same amount, so we can pick a synthetic 
    anchor date and create plausible looking dates.
    """
    years = [x[-1] for x in matches]

    if not years:
        return text, 0

    # change dates to realistic years
    sample_range = range(2008, 2020)
    # date are already in valid ranges
    if len([1 for yr in years if min_date <= yr < max_date]) == len(years):
        delta = 0
    else:
        # use document time stamp if available, 
        anchor_ts = max(years) if doc_ts is None else doc_ts
        delta = int(anchor_ts - np.random.choice(sample_range, 1)[0])

    normed = [y - delta for y in years]

    if min(normed) < min_date:
        print(delta)
        print(years)
        print(f'ERROR - normed date {min(normed)} below {min_date}')
        print("=" * 100)
        return text, 0

    for match, year in matches:
        i, j = match.span()
        s = text[i:j].replace(str(year), str(year - delta))
        if preserve_offsets:
            s = s.replace('[**', '   ').replace('**]', '   ')
        else:
            s = s.replace('[**', '%@%').replace('**]', '%@%')
        text = text[:i] + s + text[j:]

    return text, delta


def strip_phi_blinding(text, matches):
    for match, repl in matches:
        i, j = match.span()
        text = text[:i] + repl + text[j:]
    return text


def strip_over_cont(text):
    """

    :param text:
    :return:
    """
    # remove junk headers that concatenate multiple notes
    sents = []
    skip = False
    for line in text.split('\n'):
        if line.strip() == '(Over)':
            skip = True
        elif line.strip() == '(Cont)':
            skip = False
            continue
        if not skip:
            sents.append(line)
    text = '\n'.join(sents)

    return text

def preprocess(text, doc_ts = None, preserve_offsets=True):
    
    rgx = r'''\[\*\*(.+?)\*\*\]'''
    
    # If we are using annotated datasets, we have to preserve the original character offsets
    # so we can't remove any header text
    if not preserve_offsets:
        text = strip_over_cont(text)

    repls = set()
    years = set()
    
    for m in re.finditer(rgx, text, re.I):
        year = get_year(m.group(1))
        if year:
            years.add((m, year))
        else:
            rep = m.group().replace('[**', '   ').replace('**]', '   ')
            rep = rep.replace(m.group(1), m.group(1).replace(' ', '_'))
               
            # ignore numeric spans of the form [**01-04**]
            if not re.search(r'''^[0-9]{1,2}[-/][0-9]{1,2}$|^[0-9]{4}$''', rep.strip(' |')):
                rep = re.sub(r'''[0-9]''', 'X', rep)

            rep = re.sub(r'''[()]''', '_', rep)
            if not preserve_offsets:
                rep = rep.strip()
                rep = '%@%' + rep + '%@%'
                
            repls.add((m, rep))

    # replace dates
    text, tdelta = add_synthetic_dates(text, years, doc_ts=doc_ts, preserve_offsets=preserve_offsets)
    # replace other PHI tokens
    text = strip_phi_blinding(text, repls)
    if not preserve_offsets:
        text = text.replace('%@%', '')
    return text, tdelta

