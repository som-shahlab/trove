#
# MIMIC-III tools for replacing PHI tokens with realistic looking data.
# (only supports dates ATM)
#

import re
import numpy as np

def synthetic_mimic_dates(documents, min_date=1900, max_date=2020):
    """
    MIMIC data doesn't include real years in dates mentions. However,
    years are all offset by the same amount, so we can pick a synthetic anchor date
    and create plausible looking dates. 
    """
    obs = []
    valid_range = range(2008,2020)
    
    for i in range(len(documents)):
        years = []
        for s in documents[i].sentences:
            for m in re.finditer(r'''[|]{3}.+?[|]{3}''', s.text, re.I):
                if not re.search(r'''[A-Za-z]+''', m.group()):
                    d = re.search(r'''[|]{3}([1-8][0-9]{3})[-][0-9]{1,2}[-][0-9]{1,2}[|]{3}''', m.group(), re.I)
                    if d:
                        yr = int(d.group(1))
                        years.append(yr)
        
        # no years found OR all years occur within the plausible min/max date range
        if not years or len([1 for yr in years if min_date <= yr < max_date]) == len(years):
            continue
   
        # change dates to realistic years
        delta = max(years) - np.random.choice(valid_range,1)[0]
        normed = sorted([y-delta for y in years])
        
        if min(normed) < min_date:
            print(documents[i].name)
            print(delta)
            print(years)
            print(f'Doc {i} ERROR - normed date {min(normed)} below {min_date}')
            print("=" * 100)
            continue
            
        for s in documents[i].sentences:
            for m in re.finditer(r'''[|]{3}.+?[|]{3}''', s.text, re.I):
                if m and not re.search(r'''[A-Za-z]+''', m.group()):
                    d = re.search(r'''[|]{3}([1-8][0-9]{3})[-][0-9]{1,2}[-][0-9]{1,2}[|]{3}''', m.group(), re.I)
                    if d:
                        y = years.pop(0)
                        mention = d.group()
                        r_mention = mention.replace(f'{y}', f'{y - delta}')
                        # replace mentions
                        s.words = [w.replace(mention, r_mention) for w in s.words]
                        
                        
def fix_mimic_blinding(documents):
    """
    Convert MIMIC blinded tokens to form that can be recognized by datetime regular expressions. 
    """
    for i in range(len(documents)):
        for j,s in enumerate(documents[i].sentences):
            
            # no blinded tokens found in this string
            if not re.search(r'''[|]{3}.+?[|]{3}''', s.text, re.I):
                continue
            
            # if word contains a blinded token AND that token is not a date string
            # replace any extraneous numbers with Xs to prevent false date matches
            words = []
            for w in s.words:
                m = re.search(r'''[|]{3}.+?[|]{3}''', w, re.I)
                
                if m and re.search(r'''[A-Za-z]+''', m.group()):
                    # replace any numeric values with Xs 
                    substr = re.search('[0-9]+', m.group())
                    if substr:
                        num_chars = len(substr.group())
                        
                        repl = re.sub('[0-9]+','X' * num_chars, m.group())
                        w = w.replace(m.group(), repl)
                words.append(w)
            s.words = words
            
            # correct offsets for whitespace after replacement MIMIC markup chars
            fix_offsets = False
            words = []
            for w in s.words:
                m = re.search(r'''[|]{3}.+?[|]{3}''', w, re.I)
                if m:
                    fix_offsets = True
                words.append(w.replace('|||','   ') if m else w)
            s.words = words

            # fix char offsets for words that now have leading/trailing whitespace 
            if fix_offsets:
                words, abs_char_offsets = [], []
                for idx,w in zip(s.abs_char_offsets, s.words):
                    # remove any whitespace padding but leave tokens that are just whitespace
                    lpad = len(w) - len(w.lstrip(' ')) if w.strip() else 0
                    abs_char_offsets.append(idx + lpad)
                    words.append(w.strip() if len(w.strip()) > 0 else w)
                    
                s.words = words
                s.abs_char_offsets = abs_char_offsets
