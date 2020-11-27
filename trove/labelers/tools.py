import re
import codecs
import glob
import itertools
import collections
import pandas as pd
from trove.labelers.norm import lowercase
from scipy.sparse import dok_matrix, vstack, csr_matrix
from trove.labelers.umls import UMLS

def create_word_lf_mat(Xs, Ls, num_lfs):
    """
    Create word-level LF matrix from LFs indexed by sentence/word
    0 words X lfs
    1 words X lfs
    2 words X lfs
    ...

    """
    Yws = []
    for sent_i in range(len(Xs)):
        ys = dok_matrix((len(Xs[sent_i].words), num_lfs))
        for lf_i in range(num_lfs):
            for word_i, y in Ls[sent_i][lf_i].items():
                ys[word_i, lf_i] = y
        Yws.append(ys)
    return csr_matrix(vstack(Yws))

###############################################################################
#
# Load Ontologies
#
###############################################################################

def load_umls(fpath, stopwords=None, min_dict_size=500, max_tok_len=8):
    """ Load Unified Medical Language System (UMLS)

    Parameters
    ----------
    fpath
    stopwords
    min_dict_size
    max_tok_len

    Returns
    -------

    """
    transforms = {
        '*': [lowercase]
    }
    umls = UMLS(data_root=fpath,
                transforms=transforms,
                min_dict_size=min_dict_size,
                stopwords=stopwords,
                max_tok_len=max_tok_len)

    print(f'Loaded UMLS {len(umls.dictionary)} ontologies')
    return umls


def load_medispan(fpath, min_char_len=2):
    d = set()
    with open(fpath,'r') as fp:
        for line in fp:
            row = re.split(r'''\s{2,}''', line.strip()[6:])
            m = re.search(r'''(IJ|IV|IP|SC|PO|XX|NA|EX)[-A-Z0-9]+$''', row[0])
            term = row[0].replace(m.group(),'') if m else row[0]
            if len(term) >= min_char_len:
                d.add(term)
    return d


def load_chebi_ontology(filename,
                        stopwords=None,
                        ignore_case=True,
                        min_tok_len=1):
    """Chemical Entities of Biological Interest (ChEBI)
    https://www.ebi.ac.uk/chebi/

    Parameters
    ----------
    filename
    stopwords
    ignore_case
    min_tok_len

    Returns
    -------

    """
    stopwords = {} if not stopwords else stopwords
    terms = [line.strip().split("\t")[4] for line in
             codecs.open(filename, "rU", "utf-8").readlines()[1:]]
    terms = [lowercase(t) if ignore_case else t for t in terms if
             t.lower() not in stopwords]
    terms = [t for t in terms if len(t) >= min_tok_len]
    return dict.fromkeys(terms)


def load_ctd_dictionary(filename, stopwords=None):
    """Comparative Toxicogenomics Database

    Parameters
    ----------
    filename
    stopwords

    Returns
    -------

    """
    stopwords = stopwords if stopwords else {}

    d = {}
    header = ['DiseaseName', 'DiseaseID', 'AltDiseaseIDs', 'Definition',
              'ParentIDs', 'TreeNumbers', 'ParentTreeNumbers', 'Synonyms',
              'SlimMappings']

    synonyms = {}
    dnames = {}
    with open(filename ,"r") as fp:
        for i ,line in enumerate(fp):
            line = line.strip()
            if line[0] == "#":
                continue
            row = line.split("\t")
            if len(row) != 9:
                continue
            row = dict(zip(header ,row))

            synset = row["Synonyms"].strip().split("|")
            if synset:
                synonyms.update(dict.fromkeys(synset))
            term = row["DiseaseName"].strip()
            if term:
                dnames[term] = 1

    terms = {lowercase(t) for t in \
             set(list(synonyms.keys()) + list(dnames.keys())) if t}
    # filter out stopwords
    return {t for t in terms if t not in stopwords and \
            not re.search(r'''^[0-9]$''' ,t)}


def load_bioportal_dict(fname,
                        include_synset=True,
                        transforms=None,
                        stopwords=None):
    """

    Parameters
    ----------
    fname
    include_synset
    transforms
    stopwords

    Returns
    -------

    """
    transforms = [] if not transforms else transforms
    stopwords = {} if not stopwords else stopwords

    df = pd.read_csv(fname, delimiter=',', sep='"')
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]

    pref = df.preferred_label.astype(str)
    synset = itertools.chain.from_iterable(
        [t.split('|') for t in df.synonyms.astype(str)])
    termset = list(pref) + list(synset)

    d = set()
    for term in termset:
        if not term:
            continue
        for tf in transforms:
            term = tf(term)
        if term in stopwords:
            continue
        d.add(term)

    return d


def load_adam_dataset(fpath, target_concepts):
    """

    Parameters
    ----------
    fpath
    target_concepts

    Returns
    -------

    """
    synset = collections.defaultdict(set)

    with open(fpath, 'r') as fp:
        for i, line in enumerate(fp):
            if line[0] == '#':
                continue
            pref_abbrv, alt_abbrv, long_form, score, num = line.strip().split(
                "\t")

            long_form = long_form.split(':')[0]
            alt_abbrv = alt_abbrv.split(':')[0]

            if float(score) < 0.5:
                continue

            if long_form in target_concepts or lowercase(
                    long_form) in target_concepts:
                synset[pref_abbrv].add(lowercase(long_form))

    return synset


def load_specialist_abbrvs(fpath,
                           umls,
                           target_concepts=None,
                           filter_ambiguous=True):
    """

    Parameters
    ----------
    fpath
    umls
    target_concepts
    filter_ambiguous

    Returns
    -------

    """
    specialist = collections.defaultdict(list)
    with open(fpath, 'r') as fp:
        for i, line in enumerate(fp):
            row = line.strip().split("|")
            uid, abbrv, atype, uid2, term, _ = row
            if atype not in ["acronym", "abbreviation"] or not abbrv.isupper():
                continue
            # fetch all semantic types linked to this abbreviation and term
            stys = umls.get_term_stys(term)
            stys.extend(umls.get_term_stys(abbrv))

            ambiguous = False
            if filter_ambiguous:
                # threshold by class probability
                if type(filter_ambiguous) is float:
                    tmp = list(zip(*stys))[-1] if stys else []
                    tmp = {sty: tmp.count(sty) for sty in tmp}
                    wsd = {True: 0, False: 0}
                    for sty in tmp:
                        wsd[sty in target_concepts] += tmp[sty]
                    p = wsd[True] / (sum(wsd.values()) + 1e-5)

                    if p < filter_ambiguous:
                        ambiguous = True

                # or hard filter *any* ambiguous terms
                else:
                    for sab, sty in stys:
                        if sty not in target_concepts:
                            ambiguous = True
                            break

            if filter_ambiguous and ambiguous:
                continue

            for sab, sty in stys:
                if not target_concepts or sty in target_concepts:
                    specialist[abbrv].append(term)
                    break
    for abbrv in specialist:
        specialist[abbrv] = set(specialist[abbrv])
    return dict(specialist)


def load_wiki_med_abbrvs(root):
    data = collections.defaultdict(list)
    filelist = glob.glob(f'{root}/*.csv')
    for fpath in filelist:
        with open(fpath,'r') as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                row = line.split(",")
                abbrv, synset = row[0], row[1].split('\t')
                if len(synset) == 1:
                    synset = re.split('\s{2,}',synset[0])
                normed_synset = list(map(lambda x:x.lower(), synset))
                data[abbrv].extend(synset)
                data[abbrv].extend(normed_synset)
    for abbrv in data:
        data[abbrv] = set(data[abbrv])
    return dict(data)


def load_vanderbilt_datasets(filelist, class_map, cui2sty):

    vabbr = {
        1:collections.defaultdict(list),
        2:collections.defaultdict(list)
    }

    for fpath in filelist:
        df = pd.read_csv(fpath, sep='\t',header=0)
        df.abbreviation = [x.upper() for x in df.abbreviation]
        for row in df.itertuples():
            try:
                cui = row.CUI.upper().split("|")
                for c in cui:
                    if c not in cui2sty:
                        continue
                    sty = cui2sty[c]
                    label = class_map[sty] if sty in class_map else 0
                    if label == 0:
                        continue
                    vabbr[label][row.abbreviation].append(row.sense)
            except Exception as e:
                pass

    vabbr[1] = dict(vabbr[1])
    vabbr[2] = dict(vabbr[2])
    return vabbr