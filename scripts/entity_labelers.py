import re
import codecs
import functools
import itertools
import collections
import pandas as pd
from trove.utils import score_umls_ontologies
from scipy.sparse import dok_matrix, vstack, csr_matrix

from trove.labelers.labeling import (
    DictionaryLabelingFunction,
    TermMapLabelingFunction,
    OntologyLabelingFunction,
    RegexEachLabelingFunction,
    RegexLabelingFunction,
    CustomLabelingFunction
)
from trove.labelers.abbreviations import AbbrvDefsLabelingFunction
from trove.labelers.umls import load_sem_groups
from trove.labelers.tools import load_specialist_abbrvs
from trove.labelers.umls import UMLS, umls_ontology_dicts
from trove.labelers.norm import lowercase, strip_affixes

from trove.labelers.tools import (
    load_bioportal_dict,
    load_ctd_dictionary,
    load_adam_dataset,
    load_chebi_ontology
)


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


def load_umls(fpath, stopwords=None, min_dict_size=500, max_tok_len=8):
    """
    Load Unified Medical Language System
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


###############################################################################
#
# Disorders (EHR ShARe/CLEF 2014)
#
###############################################################################

def LF_single_char_rgx(s, dict_lf):
    # Search context window for abbreviated symptoms (each char is a
    # separate entity by CLEF annotation guidelines
    m = re.search(r'''\b(r/r/w|m/r/g|n/v/d|n/v|c/c/e|f/c/s|mg/r)\b''', s.text,
                  re.I)
    label = 2 if not m else 1
    char_dict = {'c', 'd', 'e', 'f', 'g', 'm', 'n', 'r', 's', 'v', 'w'}
    L = {}
    for i, tok in enumerate(s.words):
        if tok.lower() in char_dict:
            L[i] = label

    # HACK - disjoint, sparse LFs cause weird behaviors, e.g., flipping
    # signs and being completely down weighted. So we hook this LF another a
    # larger dictionary to fix the coverage problem.
    V = dict_lf(s)
    for i, y in V.items():
        L[i] = V[i]
    return L

class DisorderLabelingFunctions(object):

    def __init__(self, data_root):
        self.data_root = data_root
        self.class_map = self.load_class_map()

    def load_class_map(self):

        sem_types = list(itertools.chain.from_iterable(
            load_sem_groups(f'{self.data_root}/SemGroups.txt',
                            groupby='GUI').values())
        )

        # DISORDER semantic group
        class_map = {
            'disease_or_syndrome': 1,
            'neoplastic_process': 1,
            'injury_or_poisoning': 1,
            'sign_or_symptom': 1,
            'pathologic_function': 1,
            'finding': 0,  # *very* noisy
            'anatomical_abnormality': 1,
            'congenital_abnormality': 1,
            'acquired_abnormality': 1,
            'experimental_model_of_disease': 1,
            'mental_or_behavioral_dysfunction': 1,
            'cell_or_molecular_dysfunction': 1
        }

        # negative supervision
        class_map.update(
            {sty: 2 for sty in sem_types if sty not in class_map}
        )

        # ignore these semantic types
        ignore = [
            'amino_acid,_peptide,_or_protein',
            'gene_or_genome',
            'amino_acid_sequence',
            'nucleotide_sequence',
            'organophosphorus_compound'
        ]

        for sty in ignore:
            if sty in class_map:
                del class_map[sty]

        rm = [sty for sty in class_map if class_map[sty] == 0]
        for sty in rm:
            del class_map[sty]

        return class_map


    def lfs(self, train_sentences, top_k=5):
        """

        Parameters
        ----------
        train_sentences     unlabeled sentences for ranking ontology coverage
        top_k               use top_k ranked ontologies by coverage + rest

        Returns
        -------

        """
        sw = set(
            open(f'{self.data_root}/stopwords.txt', 'r').read().splitlines()
        )
        sw = sw.union(set([t[0].upper() + t[1:] for t in sw]))

        # Unified Medical Language System
        umls = load_umls(f'{self.data_root}/ontologies/umls2018AA.parquet', sw)

        print('(Source Vocabulary / Semantic Type) Pairs', len(umls.dictionary))
        sabs, stys = zip(*umls.dictionary.keys())
        print("Unique SAB", len(set(sabs)))
        print("Unique STY", len(set(stys)))

        # create class dictionaries (i.e., the union of all sab/stys)
        class_dictionaries = collections.defaultdict(dict)
        for sab, sty in umls.dictionary:
            if sty in self.class_map:
                class_dictionaries[self.class_map[sty]].update(
                    dict.fromkeys(umls.dictionary[(sab, sty)]))

        print("Class dictionaries")
        for label in class_dictionaries:
            print(f'y={label} n={len(class_dictionaries[label])}')

        # CTD
        fpath = f'{self.data_root}/ontologies/CTD_diseases.tsv'
        ctd_disease_dict = load_ctd_dictionary(fpath, sw)
        print(f'Loaded CTD')

        # DOID
        doid_disease_dict = load_bioportal_dict(
            f'{self.data_root}/ontologies/DOID.csv',
            transforms=[lowercase, strip_affixes],
            stopwords=sw)
        print(f'Loaded DOID')

        # HP
        hp_disease_dict = load_bioportal_dict(
            f'{self.data_root}/ontologies/HP.csv',
            transforms=[lowercase, strip_affixes],
            stopwords=sw)
        print(f'Loaded HP')

        # AutoNER
        autoner_disease_dict = [
            x.split("\t") for x in
            open(f'{self.data_root}/ontologies/autoner_BC5CDR_dict_core.txt',
                 'r').read().splitlines()
        ]
        autoner_disease_dict = set(
            [lowercase(x[-1]) for x in autoner_disease_dict if x[0] == 'Disease'])
        print(f'Loaded AutoNER')

        # SPECIALIST 2019
        fpath = f'{self.data_root}/ontologies/SPECIALIST_2019/LRABR'
        target_concepts = [sty for sty in self.class_map if self.class_map[sty] == 1]
        specialist_1 = load_specialist_abbrvs(fpath,
                                              umls,
                                              target_concepts=target_concepts,
                                              filter_ambiguous=True)
        target_concepts = [sty for sty in self.class_map if self.class_map[sty] == 2]
        specialist_2 = load_specialist_abbrvs(fpath,
                                              umls,
                                              target_concepts=target_concepts,
                                              filter_ambiguous=True)
        print(f'Loaded SPECIALIST 2019')

        # ADAM Biomedical Abbreviations Synset
        adam_synset_1 = load_adam_dataset(
            f'{self.data_root}/ontologies/ADAM/adam_database',
            class_dictionaries[1])
        adam_synset_2 = load_adam_dataset(
            f'{self.data_root}/ontologies/ADAM/adam_database',
            class_dictionaries[2])
        print(f'Loaded ADAM Biomedical Abbreviations')

        # Custom Findings Ontology (subset of the UMLS)
        findings_ontology = {}
        for term in set(
                open(f'{self.data_root}/dictionaries/findings-pos.txt',
                     'r').read().splitlines()):
            findings_ontology[term] = [0.0, 1.0]
        for term in set(
                open(f'{self.data_root}/dictionaries/findings-pos.txt',
                     'r').read().splitlines()):
            findings_ontology[term] = [1.0, 0.0]
        print(f'Loaded Findings')

        # Override terms in the UMLS
        special_cases_dict = {
            'akinetic', 'bleed', 'consolidation', 'fall', 'lesion',
            'malalignment', 'mental status', 'opacification', 'opacities',
            'swelling', 'tamponade', 'unresponsive', 'LVH', 'NAD', 'PNA',
            'DNR', 'SSCP'
        }

        # OHDSI MedTagger + Stanford SHC COVID-19 terms
        dict_covid = open(
            f'{self.data_root}/dictionaries/covid19.tsv','r'
        ).read().splitlines()

        # ---------------------------------------------------------------------
        # Initialize Ontologies
        # ---------------------------------------------------------------------
        if top_k:
            # rank source vocabs by term coverage
            scores = score_umls_ontologies(train_sentences, umls.dictionary)
            rankings = sorted(scores.items(), key=lambda x:x[-1], reverse=1)
            print(f'Matched {len(rankings)} source vocabularies')
            print(f'top_k={top_k} ({top_k/len(rankings)*100:2.1f}%)')
            sabs, _ = zip(*rankings[0:top_k])
        else:
            # merge all vocabs into a single dictionary
            print('Merge all UMLS source vocabularies')
            sabs = {}

        ontologies = umls_ontology_dicts(sabs, self.class_map, umls)

        # Fix special cases
        for name in sorted(ontologies):
            for term in special_cases_dict:
                if term in ontologies[name]:
                    ontologies[name][term] = [1., 0.]

        # ---------------------------------------------------------------------
        # Labeling Functions
        # ---------------------------------------------------------------------
        lfs = []

        # no stopwords originally provided by the guidelines
        guidelines_sw = {
            'allergy', 'allergies', 'illness', 'disease', 'syndrome',
            'drug allergies', 'significant',  'cavity', 'depressed', 'drug',
            'right', 'left', 'bilateral', 'severe', 'painful',
        }

        # guidelines https://drive.google.com/file/d/0B7oJZ-fwZvH5VmhyY3lHRFJhWkk/edit
        fpath = f'{self.data_root}/dictionaries/guidelines/clef-guidelines-disorders.txt'
        guidelines_dict = open(fpath,'r').read().splitlines()

        # =====================================================================
        #
        # Baseline LFs
        #
        # =====================================================================
        lfs.append(DictionaryLabelingFunction('LF_stopwords', sw, 2, case_sensitive=True))
        lfs.append(DictionaryLabelingFunction('LF_punct', set('!"#$%&*+,./:;<=>?@[\\]^_`{|}~'), 2))
        lfs.append(RegexEachLabelingFunction('LF_numbers_rgx', [r'''^[-]*[1-9]+[0-9]*([.][0-9]+)*$'''], 2))
        lfs.append(DictionaryLabelingFunction('LF_guidelines_1', guidelines_dict, 1))
        lfs.append(DictionaryLabelingFunction('LF_guidelines_2', guidelines_sw, 2))

        # =====================================================================
        #
        # Ontology LFs
        #
        # =====================================================================
        # UMLS ontologies
        for name in sorted(ontologies):
            lfs.append(OntologyLabelingFunction(f'LF_{name}',
                                                ontologies[name],
                                                stopwords = guidelines_sw))

        # Schwartz-Hearst abbreviations & Synonym sets
        lfs.append(AbbrvDefsLabelingFunction('LF_schwartz_hearst_abbrvs_1', class_dictionaries[1], 1))
        lfs.append(AbbrvDefsLabelingFunction('LF_schwartz_hearst_abbrvs_2', class_dictionaries[2], 2))

        # =====================================================================
        #
        # External Ontologies
        #
        # =====================================================================
        lfs.append(TermMapLabelingFunction('LF_specialist_synset_1', specialist_1, 1))
        lfs.append(TermMapLabelingFunction('LF_specialist_synset_2', specialist_2, 2))
        lfs.append(TermMapLabelingFunction('LF_adam_synset_1', adam_synset_1, 1))
        lfs.append(TermMapLabelingFunction('LF_adam_synset_2', adam_synset_2, 2))

        lfs.append(DictionaryLabelingFunction('LF_ctd', ctd_disease_dict, 1, stopwords=guidelines_sw))
        lfs.append(DictionaryLabelingFunction('LF_doid', doid_disease_dict, 1, stopwords=guidelines_sw))
        lfs.append(DictionaryLabelingFunction('LF_hp', hp_disease_dict, 1, stopwords=guidelines_sw))
        lfs.append(DictionaryLabelingFunction('LF_autoner', autoner_disease_dict, 1, stopwords=guidelines_sw))

        # =====================================================================
        #
        # Custom LFs (Regexes, Dictionaries, Heuristics)
        #
        # =====================================================================

        # Findings dictionary
        lfs.append(OntologyLabelingFunction(f'LF_findings', findings_ontology, stopwords=guidelines_sw))

        # Misc disorders
        rgx_disorders = [
            r'''(non[-]*)*calcified (nodule|granuloma|plaquing|mass[e]*)[s]*''',
            r'''(((epi|calci)[a-z]+|lung|hilar|renal|sacral|pancreas|ampullary|bladder|underlying|LV) )*mass(es)*''',
            r'''(incomplete )*((left|right)([- ]sided)*|bilateral)*\s*(bundle-branch block|pleural effusion[s]*)''',
            r'''(wide complex )*tachy(cardi[a]*c|pneic)*''',
            r'''(((radial|superior|proximal|healed|nondisplaced|open) )*((transverse process|(pubic )*rami|iliac bone|fibula|ulnar|sacral|pelvic|neck|rami|C2) )*)*fracture''',
            r'''(decreased|mechanical|positive|coarse) (upper airway|breath|bowel|heart) sound[s]*''',
            r'''((small|multi|four|three|two|one)\s*[-]\s*vessel|[1234]\s*[-]\s*vessel|coronary artery|atherosclerotic|atheromatous|LM(CA)*) disease''',
            r'''((left main|right|left)\s*coronary artery|LCMA|RCA) (disease|stenosis)''',
            
            r'''(mental status|(nonspecific )*(ST(-T)*|T)( (segment|wave))*|EKG|MS|vision|cardi[ao]pulm(onary)*|weight) change[s]*''',
            r'''(abnormal(ities|ity)*) (PAP smear|septal motion|movement|alignment|period|signal|EEG)[s]*''',
            r'''food (impaction[s]*|impacted)'''
        ]
        rgx_disorders = [re.compile(rgx, re.I) for rgx in rgx_disorders]
        lfs.append(RegexLabelingFunction('LF_disorders_rgx_1', rgx_disorders, 1))

        # Pulmonary disease
        rgx_pulmonary = [
            r'''(chronic obstructive pulmonary disease|((chronic|acute|ischemic) )*(cardio)*pulmonary (artery hypertension|vascular congestion|hyperinflation|consolidation|hypertension|infiltrate|sarcoidosis|contusion|process|effusion|disease|overload|embolism|sequelae|process|embolus|nodule|status|edema)[s]*)'''
        ]
        rgx_pulmonary = [re.compile(rgx, re.I) for rgx in rgx_pulmonary]
        lfs.append(RegexLabelingFunction('LF_pulmonary_rgx_1', rgx_pulmonary, 1))

        # Misc. abbreviations
        ambig_abbrvs_dict = {
            'MVC', 'HD', 'BMD', 'CCA', 'SEM', 'PICA', 'AKU', 'CMO', 'AT', 'ACLS',
            'WED', 'SDA', 'ABD', 'NPH', 'EXT', 'FBS', 'LCA', 'ID', 'FHL', 'INR', 'LAD',
        }
        lfs.append(DictionaryLabelingFunction('LF_abbrvs_2', ambig_abbrvs_dict, 2))

        abbrvs_dict = {
            'AH', 'APD', 'CAD', 'CKD', 'CMV', 'CP', 'CPP', 'DOE', 'GCS', 'GIB',
            'GSW', 'LBP', 'LOC', 'LVH', 'NS', 'NSR', 'OD', 'OSA',
            'PNA', 'SAH', 'SBP', 'SOB', 'UTI', 'PNA'
        }
        lfs.append(DictionaryLabelingFunction('LF_abbrvs_1', abbrvs_dict, 1))

        # spatial modifiers
        rgx_spatial = [r'''((right|left|L|R)[- ]+side[d]*|bilateral|right/left|right|left|lower|upper|R[/ ]+L|RUL|RUQ|LE) (ventricular enlargement|gastrointestinal bleed|gastrointestinal bleed|abdominal tenderness|atrial enlargement|extremity edema|disorientation|septic emboli|pneumothorax|hemiparesis|hemiparesis|infiltrates|tenderness|pneumonia|confusion|effusions|confusion|weakness|hematoma|edema)''']
        lfs.append(RegexLabelingFunction('LF_spatial_rgx_1', rgx_spatial, 1))

        # these are UMLS concepts that are not Disorder concepts so they
        # are labeled incorrectly with UMLS binary supervision
        umls_err_dict = set([
            'fall', 'bleed', 'lesion', 'lesions', 'swelling', 'akinetic',
            'tamponade', 'opacities', 'complaints', 'unresponsive',
            'malalignment', 'mental status', 'consolidation', 'opacification',
            'acute distress', 'carotid bruits', 'temperature spikes'
        ])
        lfs.append(DictionaryLabelingFunction('LF_umls_non_diso_1', umls_err_dict, 1))

        # COVID-19 Symptoms / Terms
        lfs.append(DictionaryLabelingFunction('LF_covid19_1', dict_covid, 1))


        # TODO - Investigate these more carefully. These LFs perform weirdly
        # with the label model due to known issues estimating params
        # with disjoint / sparse / unipolar LFs

        # Single character disorders
        chv = [lf for lf in lfs if lf.name == 'LF_CHV' or 'LF_OTHER'][0]
        lf_char = functools.partial(LF_single_char_rgx, dict_lf=chv)
        lfs.append(CustomLabelingFunction("LF_single_char_1", lf_char))

        # Composite symptom abbreviations
        composite_symptoms_dict = {
            'r/r/w', 'm/r/g', 'n/v/d', 'n/v', 'c/c/e', 'f/c/s',
            'R/R/W', 'M/R/G', 'N/V/D', 'N/V', 'C/C/E', 'F/C/S',
        }
        lfs.append(DictionaryLabelingFunction('LF_composite_symptoms_1', composite_symptoms_dict, 1, max_ngrams=5))

        # Medical shorthand
        med_shorthand_dict = set([
            'h/o', 's/p', "d/c'd", 'd/c', "d/c'ed", 'd/ced', 'bp', 'y/o',
            'c/b', 'i/p', 'd/t', 'n/a', 'w/o', 'w/d', 'b/l', 'c/w', 'p/w', 'w/',
            'f/u', '+/-', 'u/s', 'o/w',
        ])
        lfs.append(DictionaryLabelingFunction('LF_med_shorthand_2', med_shorthand_dict, 2, max_ngrams=5))

        print(f'Labeling Functions n={len(lfs)}')

        return lfs


###############################################################################
#
# Chemical (BC5CDR)
#
###############################################################################

class ChemicalLabelingFunctions(object):

    def __init__(self, data_root):
        self.data_root = data_root
        self.class_map = self.load_class_map()

    def load_class_map(self):
        """Class definition from UMLS semantic types (STY)

        Returns
        -------

        """
        concepts = load_sem_groups(
            f'{self.data_root}/SemGroups.txt',
            groupby='GUI'
        )
        sem_types = list(itertools.chain.from_iterable(concepts.values()))

        # CHEMICAL semantic group
        class_map = {sab: 1 for sab in concepts['CHEM']}

        # negative supervision
        class_map.update(
            {sty: 2 for sty in sem_types if sty not in class_map}
        )

        # ignore these semantic types
        ignore = [
            'biologically_active_substance',
            'gene_or_genome',
            'amino_acid_sequence',
            'nucleotide_sequence',
            'receptor',
            'amino_acid,_peptide,_or_protein',
            'pharmacologic_substance',
            'indicator,_reagent,_or_diagnostic_aid',
            'biomedical_or_dental_material',
            'immunologic_factor',
            'enzyme',
            'hormone'
        ]

        for sty in ignore:
            if sty in class_map:
                del class_map[sty]

        rm = [sty for sty in class_map if class_map[sty] == 0]
        for sty in rm:
            del class_map[sty]

        return class_map


    def lfs(self, train_sentences, top_k=10):

        print('Loading Ontologies/Dictionaries')
        print('=' * 80)
        sw = set(
            open(f'{self.data_root}/stopwords.txt', 'r').read().splitlines()
        )
        sw = sw.union(set([t[0].upper() + t[1:] for t in sw]))

        # Unified Medical Language System
        umls = load_umls(f'{self.data_root}/ontologies/umls2018AA.parquet', sw)

        print('(Source Vocabulary / Semantic Type) Pairs',
              len(umls.dictionary))
        sabs, stys = zip(*umls.dictionary.keys())
        print("Unique SAB", len(set(sabs)))
        print("Unique STY", len(set(stys)))

        # create class dictionaries (i.e., the union of all sab/stys)
        class_dictionaries = collections.defaultdict(dict)
        for sab, sty in umls.dictionary:
            if sty in self.class_map:
                class_dictionaries[self.class_map[sty]].update(
                    dict.fromkeys(umls.dictionary[(sab, sty)]))

        print("Class dictionaries")
        for label in class_dictionaries:
            print(f'y={label} n={len(class_dictionaries[label])}')

        # CTD
        fpath = f'{self.data_root}/ontologies/CTD_chemicals.tsv'
        ctd_chemical_dict = load_ctd_dictionary(fpath, sw)
        fpath = f'{self.data_root}/ontologies/CTD_diseases.tsv'
        ctd_disease_dict = load_ctd_dictionary(fpath, sw)
        ctd_onto = {lowercase(t):[1.0, 0.0]  for t in ctd_chemical_dict}
        ctd_onto.update({lowercase(t):[0.0, 1.0]  for t in ctd_disease_dict})
        print(f'Loaded CTD')

        # DOID
        doid_disease_dict = load_bioportal_dict(
            f'{self.data_root}/ontologies/DOID.csv',
            transforms=[lowercase, strip_affixes],
            stopwords=sw)
        print(f'Loaded DOID')

        # HP
        hp_disease_dict = load_bioportal_dict(
            f'{self.data_root}/ontologies/HP.csv',
            transforms=[lowercase, strip_affixes],
            stopwords=sw)
        print(f'Loaded HP')

        # AutoNER
        autoner_dict = [
            x.split("\t") for x in
            open(f'{self.data_root}/ontologies/autoner_BC5CDR_dict_core.txt',
                 'r').read().splitlines()
        ]
        autoner_chemical_dict = set(
            [lowercase(x[-1]) for x in autoner_dict if
             x[0] == 'Chemical'])
        autoner_onto = {
            lowercase(x[-1]):[1.0, 0.0] if x[0] == 'Chemical' else [0.0, 1.0]
            for x in autoner_dict
        }
        print(f'Loaded AutoNER')

        # SPECIALIST 2019
        fpath = f'{self.data_root}/ontologies/SPECIALIST_2019/LRABR'
        target_concepts = [sty for sty in self.class_map if
                           self.class_map[sty] == 1]
        specialist_1 = load_specialist_abbrvs(fpath,
                                              umls,
                                              target_concepts=target_concepts,
                                              filter_ambiguous=True)
        target_concepts = [sty for sty in self.class_map if
                           self.class_map[sty] == 2]
        specialist_2 = load_specialist_abbrvs(fpath,
                                              umls,
                                              target_concepts=target_concepts,
                                              filter_ambiguous=True)
        print(f'Loaded SPECIALIST 2019')

        # ADAM Biomedical Abbreviations Synset
        adam_synset_1 = load_adam_dataset(
            f'{self.data_root}/ontologies/ADAM/adam_database',
            class_dictionaries[1])
        adam_synset_2 = load_adam_dataset(
            f'{self.data_root}/ontologies/ADAM/adam_database',
            class_dictionaries[2])
        print(f'Loaded ADAM Biomedical Abbreviations')

        # CHEBI
        chebi_dict = load_chebi_ontology(
            f'{self.data_root}/ontologies/CHEBI/names.tsv', stopwords=sw)
        print(f'Loaded CHEBI')

        # ---------------------------------------------------------------------
        # Initialize Ontologies
        # ---------------------------------------------------------------------
        if top_k:
            # rank source vocabs by term coverage
            scores = score_umls_ontologies(train_sentences, umls.dictionary)
            rankings = sorted(scores.items(), key=lambda x:x[-1], reverse=1)
            print(f'Matched {len(rankings)} source vocabularies')
            print(f'top_k={top_k} ({top_k/len(rankings)*100:2.1f}%)')
            sabs, _ = zip(*rankings[0:top_k])
        else:
            # merge all vocabs into a single dictionary
            print('Merge all UMLS source vocabularies')
            sabs = {}

        ontologies = umls_ontology_dicts(sabs, self.class_map, umls)

        # ---------------------------------------------------------------------
        # Summarize Ontology/Dictionary Set Sizes
        # ---------------------------------------------------------------------
        resources = {
            'CHEBI': chebi_dict,
            'ADAM_1': adam_synset_1,
            'ADAM_2': adam_synset_2,
            'SPECIALIST_2019_1': specialist_1,
            'SPECIALIST_2019_2': specialist_2,
            'AutoNER': autoner_onto,
            'CTD': ctd_onto,
            'DOID': doid_disease_dict,
            'HP' : hp_disease_dict
        }
        print("Ontologies/Dictionaries")
        for name in ontologies:
            print(f'[UMLS:{name}] {len(ontologies[name])}')
        for name in resources:
            print(f'[{name}] {len(resources[name])}')
        print('=' * 80)

        lfs = []

        # Task guidelines provide example annotations
        # https://biocreative.bioinformatics.udel.edu/media/store/files/2015/bc5_CDR_data_guidelines.pdf
        # Page 6 "Do not annotate"
        #guidelines = 'cdr-chemicals-guidelines-neg.txt'
        guidelines = 'cdr-chemicals-guidelines-neg-expanded.txt'
        fpath = f'{self.data_root}/dictionaries/guidelines/{guidelines}'

        guidelines_sw = open(fpath,'r').read().splitlines()
        guidelines_sw = set(guidelines_sw).union(
            {t + 's' for t in guidelines_sw}
        )

        drug_class_dict = {
            'antibiotic',
            'antibodies',
            'antibody',
            'anticholinergic',
            'anticoagulant',
            'antiepileptic',
            'antiepileptic drug',
            'antihypertensive',
            'antihypertensive drug',
            'antineoplastic drug',
            'antioxidant',
            'antiinflammatory',
            'antiinflammatories',
            'antipsychotic',
            'cholinergic agent',
            'corticosteroid',
            'diuretic',
            'protease inhibitor',
            'receptor antagonist'
        }
        drug_class_dict.update({t+'s' for t in drug_class_dict})
        guidelines_sw = set(guidelines_sw).union(drug_class_dict)

        # page 6
        # SELECT STR FROM MRCONSO WHERE CODE in (D004967, D000928, D003276)
        special_cases = {
            'estrogen', 'estrogen receptor agonist', 'estrogenic agent',
            'estrogenic compound', 'estrogenic effect', 'antidepressant',
            'antidepressive agent', 'antidepressant drug', 'thymoanaleptics',
            'thymoleptics', 'low-dose oral contraceptive',
            'oral contraceptive', 'phasic oral contraceptive'
        }
        special_cases = special_cases.union({t + 's' for t in special_cases})

        # page 5 of bc5_CDR_data_guidelines
        guidelines_dict = {
            'amino acid', 'fatty acid', 'steroid', 'saturated fatty acid',
            'glucose', 'sucrose', 'angiotensin II', 'angiotensin ii', 'ATP',
            'cAMP', 'polyethylene glycol',
            'ethanolic extract of daucus carota seed', 'levodopa', 'carbidopa',
            'grape seed proanthocyanidin extract', 'DCE', 'nitric oxide', 'NO'
        }
        guidelines_dict.update({t + 's' for t in guidelines_dict})
        elements_dict = {'Ca', 'Fe', 'Li', 'K', 'O2'}
        guidelines_dict.update(elements_dict)

        # =======================================================================
        #
        # Baseline LFs
        #
        # =======================================================================
        MAX_NGRAMS = 8

        # guideline labeling functions
        lfs.append(DictionaryLabelingFunction('LF_guidelines_1', guidelines_dict, 1, max_ngrams=MAX_NGRAMS))
        lfs.append(DictionaryLabelingFunction('LF_guidelines_2', guidelines_sw, 2, max_ngrams=MAX_NGRAMS))
        lfs.append(DictionaryLabelingFunction('LF_special_cases', special_cases, 1, max_ngrams=MAX_NGRAMS))

        # stopwords, numbers, and punctuation.
        lfs.append(DictionaryLabelingFunction('LF_stopwords', sw, 2, max_ngrams=MAX_NGRAMS, case_sensitive=True))

        # =======================================================================
        #
        # UMLS Ontology LFs
        #
        # =======================================================================

        # UMLS ontologies
        for name in sorted(ontologies):
            lfs.append(OntologyLabelingFunction(f'LF_{name}',
                                                ontologies[name],
                                                max_ngrams=MAX_NGRAMS,
                                                stopwords=guidelines_sw))

        # Schwartz-Hearst abbreviations + UMLS
        lfs.append(AbbrvDefsLabelingFunction('LF_schwartz_hearst_abbrvs_1',
                                             class_dictionaries[1], 1,
                                             stopwords=guidelines_sw))
        lfs.append(AbbrvDefsLabelingFunction('LF_schwartz_hearst_abbrvs_2',
                                             class_dictionaries[2], 2))

        # =======================================================================
        #
        # External Ontologies
        #
        # =======================================================================

        # Synonym sets
        lfs.append(TermMapLabelingFunction('LF_specialist_synset_1', specialist_1, 1, stopwords=guidelines_sw))
        lfs.append(TermMapLabelingFunction('LF_specialist_synset_2', specialist_2, 2))
        lfs.append(TermMapLabelingFunction('LF_adam_synset_1', adam_synset_1, 1, stopwords=guidelines_sw))
        lfs.append(TermMapLabelingFunction('LF_adam_synset_2', adam_synset_2, 2))

        # Dictionaries
        lfs.append(DictionaryLabelingFunction('LF_chebi', chebi_dict, 1, max_ngrams=MAX_NGRAMS, stopwords=guidelines_sw))
        lfs.append(DictionaryLabelingFunction('LF_ctd', ctd_chemical_dict, 1, max_ngrams=MAX_NGRAMS, stopwords=guidelines_sw))
        lfs.append(DictionaryLabelingFunction('LF_doid', doid_disease_dict, 2, max_ngrams=MAX_NGRAMS, stopwords=guidelines_sw))
        lfs.append(DictionaryLabelingFunction('LF_hp', hp_disease_dict, 2, max_ngrams=MAX_NGRAMS, stopwords=guidelines_sw))
        lfs.append(DictionaryLabelingFunction('LF_autoner', autoner_chemical_dict, 1, max_ngrams=MAX_NGRAMS, stopwords=guidelines_sw))

        # =======================================================================
        #
        # Custom LFs (Regexes, Dictionaries, Heuristics)
        #
        # =======================================================================

        # enzymes, peptides, proteins of the from "*rase" are not annotated as chemicals
        proteins_rgx = [
            r'''\b([A-Za-z0-9]+?[rlntd]ase[s]*)\b''',
            r'''[A-Za-z0-9]+ factor[s]*''',
            r'''angiotensinogen'''
        ]
        lfs.append(RegexLabelingFunction('LF_proteins_rgx', proteins_rgx, 2))

        # some supplements aren't annotated
        suppliments_dict = {'guarana', 'panax ginseng', 'ginseng'}
        lfs.append(DictionaryLabelingFunction('LF_herbal_suppliments', suppliments_dict, 2,
                                             max_ngrams=MAX_NGRAMS, stopwords=guidelines_sw))

        # classes of anti* drugs (antibiotics, antiepiletics)
        lfs.append(RegexLabelingFunction('LF_anti_rgx', [r'''\b(anti[a-z]+)\b'''], 2))

        # peptides and proteins with less than 15 amino acids ARE annotated
        # TODO - see if amino acid counts are in CHEBI or another chemical KB
        amino_acid_rgxs = [
            r'''(angiotensin([- ]ii)*)''',
            r'''(u[- ]ii|urotensin[- ]ii)''', # a peptide (11 amino acids in humans)
            r'''bradykinin''',                # 9 amino acids
            r'''(d[- ]pen(icillamine)*)'''    #
        ]
        lfs.append(RegexLabelingFunction('LF_amino_acids_rgx', amino_acid_rgxs, 1))

        # misc chemicals
        rgxs = [
            r'''(corticosteroid[s]*|appetite[- ]suppressant[s]*|oral[- ]contraceptive[s]*)''',
            r'''(calcium|cacl[(]2[)])''',
            r'''([l][- ](glutathione|arginine))'''
        ]
        lfs.append(RegexLabelingFunction('LF_misc_rgx', rgxs, 1))

        # abbreviation errors from the non-UMLS resources
        abbrvs_dict = {
            'mRNA', 'Pgp', 'TMA', 'ATT', 'BMC', 'BMCs', 'CR', 'AED', 'CO',
            'HR', 'RAS', 'CN', 'ALT', 'BUN', 'CPK', 'HEM', 'TAA', 'LS'}
        lfs.append(DictionaryLabelingFunction('LF_abbrv_errs', abbrvs_dict, 2, max_ngrams=1))

        # ignore drug class mentions
        lfs.append(DictionaryLabelingFunction('LF_drug_class', drug_class_dict, 2, max_ngrams=1))

        # =====================================================================
        #
        # Labeled Training Examples as Dictionary
        #
        # =====================================================================
        # This assumes we're built a dictionary by examining/labeling data
        # lfs.append(DictionaryLabelingFunction('LF_training_entities', train_dict, 1,
        #                                      max_ngrams=MAX_NGRAMS, stopwords=guidelines_sw))

        print(f'Labeling Functions n={len(lfs)}')
        return lfs


# =======================================================================
#
# Drug (i2b2)
#
# =======================================================================
from trove.labelers.spans.negex import NegEx
from trove.labelers.spans.taggers import (
    get_left_span,
    get_right_span,
    token_distance,
    match_regex
)


class i2b2DrugLabelingFunctions(object):
    """
    The 2009 i2b2 medication dataset combines multiple layers of annotation
    logic, which poses challenges for dictionary-based tagging. Guidelines
    dictate:
    - Only label medications the patient is currently taking
    - Do not label medication allergies

    We use a more complex templated labeling function that allows us to remap
    span labels using another heuristic.

    """
    def __init__(self, data_root):
        self.data_root = data_root
        self.class_map = self.load_class_map()

    def load_class_map(self):
        pass


    def lfs(self, train_sentences, top_k=10):

        #
        # Supervision Sources
        #
        print('Loading Ontologies/Dictionaries')
        print('=' * 80)
        sw = set(
            open(f'{self.data_root}/stopwords.txt', 'r').read().splitlines()
        )
        sw = sw.union(set([t[0].upper() + t[1:] for t in sw]))

        # Unified Medical Language System
        umls = load_umls(f'{self.data_root}/ontologies/umls2018AA.parquet', sw)

        print('(Source Vocabulary / Semantic Type) Pairs',
              len(umls.dictionary))
        sabs, stys = zip(*umls.dictionary.keys())
        print("Unique SAB", len(set(sabs)))
        print("Unique STY", len(set(stys)))

        # create class dictionaries (i.e., the union of all sab/stys)
        class_dictionaries = collections.defaultdict(dict)
        for sab, sty in umls.dictionary:
            if sty in self.class_map:
                class_dictionaries[self.class_map[sty]].update(
                    dict.fromkeys(umls.dictionary[(sab, sty)]))

        print("Class dictionaries")
        for label in class_dictionaries:
            print(f'y={label} n={len(class_dictionaries[label])}')

        # CTD
        fpath = f'{self.data_root}/ontologies/CTD_chemicals.tsv'
        ctd_chemical_dict = load_ctd_dictionary(fpath, sw)
        fpath = f'{self.data_root}/ontologies/CTD_diseases.tsv'
        ctd_disease_dict = load_ctd_dictionary(fpath, sw)
        ctd_onto = {lowercase(t): [1.0, 0.0] for t in ctd_chemical_dict}
        ctd_onto.update({lowercase(t): [0.0, 1.0] for t in ctd_disease_dict})
        print(f'Loaded CTD')

        # DOID
        doid_disease_dict = load_bioportal_dict(
            f'{self.data_root}/ontologies/DOID.csv',
            transforms=[lowercase, strip_affixes],
            stopwords=sw)
        print(f'Loaded DOID')

        # HP
        hp_disease_dict = load_bioportal_dict(
            f'{self.data_root}/ontologies/HP.csv',
            transforms=[lowercase, strip_affixes],
            stopwords=sw)
        print(f'Loaded HP')

        # AutoNER
        autoner_dict = [
            x.split("\t") for x in
            open(f'{self.data_root}/ontologies/autoner_BC5CDR_dict_core.txt',
                 'r').read().splitlines()
        ]
        autoner_chemical_dict = set(
            [lowercase(x[-1]) for x in autoner_dict if
             x[0] == 'Chemical'])
        autoner_onto = {
            lowercase(x[-1]): [1.0, 0.0] if x[0] == 'Chemical' else [0.0, 1.0]
            for x in autoner_dict
        }
        print(f'Loaded AutoNER')

        # SPECIALIST 2019
        fpath = f'{self.data_root}/ontologies/SPECIALIST_2019/LRABR'
        target_concepts = [sty for sty in self.class_map if
                           self.class_map[sty] == 1]
        specialist_1 = load_specialist_abbrvs(fpath,
                                              umls,
                                              target_concepts=target_concepts,
                                              filter_ambiguous=True)
        target_concepts = [sty for sty in self.class_map if
                           self.class_map[sty] == 2]
        specialist_2 = load_specialist_abbrvs(fpath,
                                              umls,
                                              target_concepts=target_concepts,
                                              filter_ambiguous=True)
        print(f'Loaded SPECIALIST 2019')

        # ADAM Biomedical Abbreviations Synset
        adam_synset_1 = load_adam_dataset(
            f'{self.data_root}/ontologies/ADAM/adam_database',
            class_dictionaries[1])
        adam_synset_2 = load_adam_dataset(
            f'{self.data_root}/ontologies/ADAM/adam_database',
            class_dictionaries[2])
        print(f'Loaded ADAM Biomedical Abbreviations')

        # CHEBI
        chebi_dict = load_chebi_ontology(
            f'{self.data_root}/ontologies/CHEBI/names.tsv', stopwords=sw)
        print(f'Loaded CHEBI')

        # ---------------------------------------------------------------------
        # Initialize Ontologies
        # ---------------------------------------------------------------------
        scores = score_umls_ontologies(train_sentences, umls.dictionary)
        rankings = sorted(scores.items(), key=lambda x: x[-1], reverse=1)

        print(f'Matched {len(rankings)} ontologies')
        print(f'top_k={top_k} ({top_k / len(rankings) * 100:2.1f}%)')
        sabs, _ = zip(*rankings[0:top_k])

        ontologies = umls_ontology_dicts(sabs, self.class_map, umls)

        # ---------------------------------------------------------------------
        # Summarize Ontology/Dictionary Set Sizes
        # ---------------------------------------------------------------------
        resources = {
            'CHEBI': chebi_dict,
            'ADAM_1': adam_synset_1,
            'ADAM_2': adam_synset_2,
            'SPECIALIST_2019_1': specialist_1,
            'SPECIALIST_2019_2': specialist_2,
            'AutoNER': autoner_onto,
            'CTD': ctd_onto,
            'DOID': doid_disease_dict,
            'HP': hp_disease_dict
        }
        print("Ontologies/Dictionaries")
        for name in ontologies:
            print(f'[UMLS:{name}] {len(ontologies[name])}')
        for name in resources:
            print(f'[{name}] {len(resources[name])}')
        print('=' * 80)

        #
        # Labeling Functions
        #
        negex = NegEx(data_root=f'{self.data_root}/negex/')

        def negex_definite_left(span):
            text = get_left_span(span, span.sentence, window=2).text
            rgx = negex.rgxs['definite']['left']
            return True if rgx.search(text) else False

        def negex_definite_right(span):
            text = get_right_span(span, span.sentence, window=2).text
            rgx = negex.rgxs['definite']['right']
            return True if rgx.search(text) else False

        reject_headers = {
            'SOCIAL HISTORY', 'ALLERGY', 'ALLERGIES', 'CC', 'DIET',
             'NEUROLOGICAL', 'LABORATORY DATA', 'CHIEF COMPLAINT',
             'FAMILY HISTORY', 'EKG', 'NEURO', 'PAST SURGICAL HISTORY',
             'FAMILY HISTORY', 'ALLG', 'LABORATORY', 'PREOPERATIVE LABS',
             'LABS'}
        header_rgx = r'''^({})[:]'''.format(
            '|'.join(sorted(reject_headers, key=len, reverse=1))
        )

        guidelines_sw = {'tobacco', 'alcohol', 'water'}

        # custom stopwords
        # These prevent biases from the main ontologies using the independent
        # Snorkel model. For example, "blood pressure medications",
        # blood pressure is a negative concept in all UMLS ontologies and we
        # don't capture the chain dep linking it to medications by default.
        guidelines_sw = guidelines_sw.union({'NAD', 'nad', 'fat', 'sugar', 'PCP', 'pcp',
                                             'chemo', 'chemotherapy', 'lipase', 'amylase',
                                             'smoke', 'mg', 'NPH', 'nph', 'KCL', 'kcl', 'INH',
                                             'calcium', 'magnesium', 'iron', 'potassium',
                                             'glucose', 'phosphate', 'electrolyte', 'salt',
                                             'meds', 'medications', 'medication', 'drug', 'gentle',
                                             'tomorrow', 'dose', 'unknown', 'air', 'toxin', 'prevent',
                                             'increase', 'subcutaneous', 'intravenous', 'diurese',
                                             'pain', 'blood pressure', 'oral', 'coated', 'enteric',
                                             'topical', 'immediate', 'release', 'slow'
                                             })

        # Common lab values (i.e., not drugs)
        labs = {
            'hemoglobin', 'sodium', 'bicarbonate', 'phosphorus', 'magnesium', 'lactate', 'PCO2', 'white count', 'lipase',
            'platelets', 'alk phos', 'white blood cell count', 'phosphate', 'MB', 'PT', 'serum osms', 'AST', 'amylase',
            'CO2', 'potassium', 'carbon dioxide', 'creatinine', 'glucose', 'troponin', 'PTT', 'white blood cell[s]*',
            'BUN', 'calcium', 'WBC', 'chloride', 'bicarb', 'ALT', 'O2 saturation', 'beta hydroxybutyrate', 'anion gap',
            'TSH', 'lipase', 'PO2', 'platelet count', 'CK', 'hematocrit', 'BNP', 'ABG', 'T[.]* bili', 'INR', 'ABG T',
            '[TD]-bili', 'T-bili', 'D-bili'
        }

        guidelines_sw = guidelines_sw.union(set(labs))
        guidelines_sw = guidelines_sw.union({t + 's' for t in guidelines_sw})

        MAX_NGRAMS = 8

        # Preliminary.Annotation.Guidelines.6.12.pdf
        guidelines_dict = open(f'{self.data_root}/dictionaries/i2b2-guidelines-drugs.txt','r').read().splitlines()
        guidelines_dict = set(guidelines_dict).union({t.lower() for t in guidelines_dict})

        lfs = []

        # =======================================================================
        #
        # Baseline LFs
        #
        # =======================================================================

        lfs.append(DictionaryLabelingFunction('LF_stopwords', sw, 2, case_sensitive=True))
        lfs.append(DictionaryLabelingFunction('LF_punct', set('!"#$%*+,./:;<=>?@[\\]^_`{|}~'), 2))
        lfs.append(RegexEachLabelingFunction('LF_numbers_rgx', [r'''^[-]*[1-9]+[0-9]*([.][0-9]+)*$'''], 2))
        lfs.append(DictionaryLabelingFunction('LF_guidelines', guidelines_dict, 1))

        # =======================================================================
        #
        # UMLS Ontology LFs
        #
        # =======================================================================

        # Guideline only patterns
        # slot_rgxs = [
        #     '{} (nebulizer|solution|cream|tablet)[s]*',
        #     '{} [(]\s*{}\s*[)]',
        #     '{} [(]\s*{} (and|[/-]) {}\s*[)]',
        #     '{} [(]\s*([A-Za-z0-9-\\\]+\s*){{1,3}}\s*[)]',
        #     '[A-Za-z0-9-]+ [(]\s*{}\s*[)]',
        #     '{}\s*[)]',
        #     '[(]\s*{}'
        # ]

        # expanded patterns
        slot_rgxs = [
            '{} (nebulizer|ointment|inhaler|solution|cream|paste|tablet|syrup|(nitro)*spray|patch|elixer|neb|(([1-9][%] )*(topical )*)*(powder|cream|patch))[s]*',
            '{} (extended|sustained|immediate|immed[.]*|slow) (release|releas|rel[.]*)\s*[)]*',
            '{} enteric coated',
            '{} [(]\s*{}\s*[)]',
            '{} [(]\s*{} (and|[/-]) {}\s*[)]',
            '{} [(]\s*([A-Za-z0-9-\\\]+\s*){{1,3}}\s*[)]',
            '[A-Za-z0-9-]+ [(]\s*{}\s*[)]',
            '{}\s*[)]',
            '[(]\s*{}'
        ]

        def invert_label(span):
            """
            If mention is any one of these cases, it is not labeled as per i2b2 guidelines:
            - The medication was not actually perscribed or taken by the patient
              e.g., negation, family member meds, headers.
            - The medication is listed as an allergy
            """
            regexes = [
                r'''\b(allergies[:]*|allergy|allergic to|allergic reaction)\b''',
                r'''\b(((grand)*(mother|father)|grand(m|p)a)([']*s)*|((parent|(daught|sist|broth)er|son|cousin)([']*s)*))\b'''
            ]
            # reject headers
            if re.search(header_rgx, span.sentence.text):
                return True

            # reject allergies
            left = get_left_span(span)
            for rgx in regexes:
                match = match_regex(rgx, left)
                if match:
                    d = token_distance(match, span)
                    if d <= 50:
                        return True

            # reject negated entities
            if negex_definite_left(span):
                return True
            return False

            # UMLS ontologies
            for name in sorted(ontologies):
                lfs.append(OntologyLabelingFunction(f'LF_{name}',
                                                    ontologies[name],
                                                    max_ngrams=MAX_NGRAMS,
                                                    stopwords=guidelines_sw))

            # for name in sorted(ontologies):
            #     lf = SlotFilledOntologyLabelingFunction(f'LF_{name}',
            #                                             ontologies[name],
            #                                             max_ngrams=MAX_NGRAMS,
            #                                             slot_patterns=slot_rgxs,
            #                                             stopwords=guidelines_sw,
            #                                             span_rule=invert_label)
            #     lfs.append(lf)

            # Schwartz-Hearst abbreviations
            lfs.append(AbbrvDefsLabelingFunction('LF_schwartz_hearst_abbrvs_1', class_dictionaries[1], 1))
            lfs.append(AbbrvDefsLabelingFunction('LF_schwartz_hearst_abbrvs_2', class_dictionaries[2], 2))

            # =======================================================================
            #
            # External Ontologies
            #
            # =======================================================================

            # Synonym sets
            lfs.append(TermMapLabelingFunction('LF_specialist_synset_1', specialist_1, 1))
            lfs.append(TermMapLabelingFunction('LF_specialist_synset_2', specialist_2, 2))
            lfs.append(TermMapLabelingFunction('LF_adam_synset_1', adam_synset_1, 1))
            lfs.append(TermMapLabelingFunction('LF_adam_synset_2', adam_synset_2, 2))

            lfs.append(DictionaryLabelingFunction('LF_ctd', ctd_disease_dict, 1, max_ngrams=MAX_NGRAMS, stopwords=guidelines_sw))
            lfs.append(DictionaryLabelingFunction('LF_doid', doid_disease_dict, 2, max_ngrams=MAX_NGRAMS, stopwords=guidelines_sw))
            lfs.append(DictionaryLabelingFunction('LF_hp', hp_disease_dict, 2, max_ngrams=MAX_NGRAMS, stopwords=guidelines_sw))

            # =======================================================================
            #
            # Custom LFs (Regexes, Dictionaries, Heuristics)
            #
            # =======================================================================

            # NOTE: Custom LFs for this task are more complicated due to i2b2's
            # drug definitions, which includes somewhat ad hoc inclusion criteria
            # of modes, as well as *not* annotating negated, allergy mentions

            # Illicit drugs
            illicit_drugs_dict = open('../data/dictionaries/illicit_drugs.txt','r').read().splitlines()
            lfs.append(DictionaryLabelingFunction('LF_illicit_drugs', illicit_drugs_dict, 2, case_sensitive=True))

            # Over-the-counter drugs
            rgx_otc = re.compile(r'''\b(mouth wash|eye drop|toe cream|stool softener)[es]*\b''')
            lfs.append(RegexLabelingFunction('LF_otc_rgx', [rgx_otc], 1))

            # Insulin
            rgx_insulin_pos = r'''(lantus [(] insulin glargine [)]|NPH\s*(with)*insulin|insulin (NPH|regular)*\s*(aspart|human)|insulin(\s+(NPH|lispro|regular|regimen|drip|lente|[0-9]+/[0-9]+))*|(((NPH )*humulin|regular|human|lantus|NPH) )*insulin)'''
            lfs.append(RegexLabelingFunction('LF_insulin_rgx_1', [rgx_insulin_pos], 1))

            # Not insulin mentions
            rgx_insulin_neg = r'''((non[-]*)*insulin[- ]dependent( diabetes mellitus)*|non[- ]*insulin|insulin sliding scale)'''
            lfs.append(RegexLabelingFunction('LF_insulin_rgx_2', [rgx_insulin_neg], 2))

            # Misc medications
            rgx_drugs = [
                r'''(KCL (IMMEDIATE|SLOW) (RELEASE|REL[.]*)|asa)''',
                r'''(chemotherapy|chemo)( regimen[s]*)*''',
                r'''(red blood|red blood cell|rbc|platelet) transfusion[s]*''',
                r'''(b|beta)[-]*blocker[s]*''',
                r'''(cardiac|pain|copd|(blood\s*)*pressure|oral|outpatient|home|these|your|pressor|pressure) (med(ication|icine)*[s]*)'''
            ]
            rgx_drugs = [re.compile(rgx, re.I) for rgx in rgx_drugs]
            lfs.append(RegexLabelingFunction('LF_drugs_rgx', rgx_drugs, 1))

            # Vitamins
            vit_rgx_1 = re.compile(r'''(vitamin|vit[.]*)[s]*[- ][abcde][-]*[1-3]*''', re.I)
            lfs.append(RegexLabelingFunction('LF_vitamins_rgx_1', [vit_rgx_1], 1))

            # Negated vitamins
            vit_rgx_2 = re.compile(r'''no ([A-Za-z]+\s*){1,2}(vitamin|vit[.]*)[s]*[- ][abcdek][-]*[1-3]*''', re.I)
            lfs.append(RegexLabelingFunction('LF_vitamins_rgx_2', [vit_rgx_2], 2))

            # Section Headers
            lfs.append(RegexLabelingFunction('LF_header_rgx', [header_rgx], 2))

            # Lab values e.g., 'phosphate 0.8'
            labs_rgx = r'''((?:{})(?: (?:was|of|at))* [0-9]{{1,3}}(?:[.][0-9]+)*[%]*)'''.format('|'.join(sorted(labs, key=len, reverse=1)))
            lfs.append(RegexLabelingFunction('LF_labs_rgx', [labs_rgx], 2))

            # Generic mentions of "medications"
            meds_rgx = r'''((?!(cardiac|pain|copd|(blood\s*)*pressure|oral|narcotic pain|outpatient|home|these|your|pressor|pressure)\s*)(medication[s]*))'''
            lfs.append(RegexLabelingFunction('LF_meds_rgx', [re.compile(meds_rgx, re.I)], 2))

            # Drug/chemical mention but not as a prescribed drug
            non_drug_rgx = r'''(potassium chloride policy|(cardiac|adenosine|guanosine)[- ]*mibi|mibi)'''
            lfs.append(RegexLabelingFunction('LF_non_meds_rgx', [re.compile(non_drug_rgx, re.I)], 2))

            lf_names = [lf.name for lf in lfs]
            print(f'Labeling Functions n={len(lfs)}')


# =======================================================================
#
# Disease (BC5CDR)
#
# =======================================================================
class DiseaseLabelingFunctions(object):

    def __init__(self, data_root):
        self.data_root = data_root
        self.class_map = self.load_class_map()

    def load_class_map(self):
        concepts = load_sem_groups(
            f'{self.data_root}/SemGroups.txt',
            groupby='GUI'
        )
        sem_types = list(itertools.chain.from_iterable(concepts.values()))

        # DISORDER semantic group
        class_map = {
            'disease_or_syndrome': 1,
            'neoplastic_process': 1,
            'injury_or_poisoning': 1,
            'sign_or_symptom': 1,
            'pathologic_function': 1,
            'finding': 0,  # *very* noisy
            'anatomical_abnormality': 1,
            'congenital_abnormality': 1,
            'acquired_abnormality': 1,
            'experimental_model_of_disease': 1,
            'mental_or_behavioral_dysfunction': 1,
            'cell_or_molecular_dysfunction': 1
        }

        # negative supervision
        class_map.update(
            {sty: 2 for sty in sem_types if sty not in class_map}
        )

        # ignore these semantic types
        ignore = []

        for sty in ignore:
            if sty in class_map:
                del class_map[sty]

        rm = [sty for sty in class_map if class_map[sty] == 0]
        for sty in rm:
            del class_map[sty]

        return class_map

    def lfs(self, train_sentences, top_k=10):

        sw = set(
            open(f'{self.data_root}/stopwords.txt', 'r').read().splitlines()
        )
        sw = sw.union(set([t[0].upper() + t[1:] for t in sw]))

        # Unified Medical Language System
        umls = load_umls(f'{self.data_root}/ontologies/umls2018AA.parquet', sw)

        print('(Source Vocabulary / Semantic Type) Pairs',
              len(umls.dictionary))
        sabs, stys = zip(*umls.dictionary.keys())
        print("Unique SAB", len(set(sabs)))
        print("Unique STY", len(set(stys)))

        # create class dictionaries (i.e., the union of all sab/stys)
        class_dictionaries = collections.defaultdict(dict)
        for sab, sty in umls.dictionary:
            if sty in self.class_map:
                class_dictionaries[self.class_map[sty]].update(
                    dict.fromkeys(umls.dictionary[(sab, sty)]))

        print("Class dictionaries")
        for label in class_dictionaries:
            print(f'y={label} n={len(class_dictionaries[label])}')

        # CTD
        fpath = f'{self.data_root}/ontologies/CTD_diseases.tsv'
        ctd_disease_dict = load_ctd_dictionary(fpath, sw)
        print(f'Loaded CTD')

        # DOID
        doid_disease_dict = load_bioportal_dict(
            f'{self.data_root}/ontologies/DOID.csv',
            transforms=[lowercase, strip_affixes],
            stopwords=sw)
        print(f'Loaded DOID')

        # HP
        hp_disease_dict = load_bioportal_dict(
            f'{self.data_root}/ontologies/HP.csv',
            transforms=[lowercase, strip_affixes],
            stopwords=sw)
        print(f'Loaded HP')

        # AutoNER
        autoner_disease_dict = [
            x.split("\t") for x in
            open(f'{self.data_root}/ontologies/autoner_BC5CDR_dict_core.txt',
                 'r').read().splitlines()
        ]
        autoner_disease_dict = set(
            [lowercase(x[-1]) for x in autoner_disease_dict if
             x[0] == 'Disease'])
        print(f'Loaded AutoNER')

        # SPECIALIST 2019
        fpath = f'{self.data_root}/ontologies/SPECIALIST_2019/LRABR'
        target_concepts = [sty for sty in self.class_map if
                           self.class_map[sty] == 1]
        specialist_1 = load_specialist_abbrvs(fpath,
                                              umls,
                                              target_concepts=target_concepts,
                                              filter_ambiguous=True)
        target_concepts = [sty for sty in self.class_map if
                           self.class_map[sty] == 2]
        specialist_2 = load_specialist_abbrvs(fpath,
                                              umls,
                                              target_concepts=target_concepts,
                                              filter_ambiguous=True)
        print(f'Loaded SPECIALIST 2019')

        # ADAM Biomedical Abbreviations Synset
        adam_synset_1 = load_adam_dataset(
            f'{self.data_root}/ontologies/ADAM/adam_database',
            class_dictionaries[1])
        adam_synset_2 = load_adam_dataset(
            f'{self.data_root}/ontologies/ADAM/adam_database',
            class_dictionaries[2])
        print(f'Loaded ADAM Biomedical Abbreviations')

        # Custom Findings Ontology (subset of the UMLS)
        findings_ontology = {}
        for term in set(
                open(f'{self.data_root}/dictionaries/findings-pos.txt',
                     'r').read().splitlines()):
            findings_ontology[term] = [0.0, 1.0]
        for term in set(
                open(f'{self.data_root}/dictionaries/findings-pos.txt',
                     'r').read().splitlines()):
            findings_ontology[term] = [1.0, 0.0]
        print(f'Loaded Findings')

        # Override terms in the UMLS
        special_cases_dict = {
            'akinetic', 'bleed', 'consolidation', 'fall', 'lesion',
            'malalignment', 'mental status', 'opacification', 'opacities',
            'swelling', 'tamponade', 'unresponsive', 'LVH', 'NAD', 'PNA',
            'DNR', 'SSCP'
        }

        # ---------------------------------------------------------------------
        # Initialize Ontologies
        # ---------------------------------------------------------------------
        if top_k:
            # rank source vocabs by term coverage
            scores = score_umls_ontologies(train_sentences, umls.dictionary)
            rankings = sorted(scores.items(), key=lambda x:x[-1], reverse=1)
            print(f'Matched {len(rankings)} source vocabularies')
            print(f'top_k={top_k} ({top_k/len(rankings)*100:2.1f}%)')
            sabs, _ = zip(*rankings[0:top_k])
        else:
            # merge all vocabs into a single dictionary
            print('Merge all UMLS source vocabularies')
            sabs = {}

        ontologies = umls_ontology_dicts(sabs, self.class_map, umls)

        # Fix special cases
        for name in sorted(ontologies):
            for term in special_cases_dict:
                if term in ontologies[name]:
                    ontologies[name][term] = [1., 0.]

        lfs = []

        # Task guidelines provide example annotations
        # https://biocreative.bioinformatics.udel.edu/media/store/files/2015/bc5_CDR_data_guidelines.pdf

        #guidelines = 'cdr-diseases-guidelines-neg.txt'
        guidelines = 'cdr-diseases-guidelines-neg-expanded.txt'
        fpath = f'{self.data_root}/dictionaries/guidelines/{guidelines}'

        guidelines_sw = open(fpath, 'r').read().splitlines()
        guidelines_sw = set(guidelines_sw).union(
            {t + 's' for t in guidelines_sw}
        )
        fpath = f'{self.data_root}/dictionaries/guidelines/cdr-guidelines-diseases.txt'
        guidelines_dict = open(fpath, 'r').read().splitlines()

        # =====================================================================
        #
        # Baseline LFs
        #
        # =====================================================================
        lfs.append(DictionaryLabelingFunction('LF_stopwords', sw, 2, case_sensitive=True))
        lfs.append(DictionaryLabelingFunction('LF_punct',set('!"#$%&*+,./:;<=>?@[\\]^_`{|}~'), 2))
        lfs.append(RegexEachLabelingFunction('LF_numbers_rgx', [r'''^[-]*[1-9]+[0-9]*([.][0-9]+)*$'''], 2))

        lfs.append(DictionaryLabelingFunction('LF_guidelines_1', guidelines_dict, 1))
        lfs.append(DictionaryLabelingFunction('LF_guidelines_2', guidelines_sw, 2))

        # =====================================================================
        #
        # UMLS Ontology LFs
        #
        # =====================================================================
        # UMLS ontologies
        for name in sorted(ontologies):
            lfs.append(OntologyLabelingFunction(f'LF_{name}',
                                                ontologies[name],
                                                stopwords=guidelines_sw,
                                                max_ngrams=5))

        # Schwartz-Hearst Abbreviations & Synonym Sets
        lfs.append(AbbrvDefsLabelingFunction('LF_schwartz_hearst_abbrvs_1', class_dictionaries[1], 1, stopwords=guidelines_sw))
        lfs.append(AbbrvDefsLabelingFunction('LF_schwartz_hearst_abbrvs_2', class_dictionaries[2], 2, stopwords=guidelines_sw))

        # =======================================================================
        #
        # External Ontologies
        #
        # =======================================================================
        lfs.append(TermMapLabelingFunction('LF_specialist_synset_1', specialist_1, 1, stopwords=guidelines_sw))
        lfs.append(TermMapLabelingFunction('LF_specialist_synset_2', specialist_2, 2, stopwords=guidelines_sw))
        lfs.append(TermMapLabelingFunction('LF_adam_synset_1', adam_synset_1, 1, stopwords=guidelines_sw))
        lfs.append(TermMapLabelingFunction('LF_adam_synset_2', adam_synset_2, 2, stopwords=guidelines_sw))

        # Misc. Lexicons / Ontologies, e.g., https://bioportal.bioontology.org/
        lfs.append(DictionaryLabelingFunction('LF_ctd', ctd_disease_dict, 1, stopwords=guidelines_sw))
        lfs.append(DictionaryLabelingFunction('LF_doid', doid_disease_dict, 1, stopwords=guidelines_sw))
        lfs.append(DictionaryLabelingFunction('LF_hp', hp_disease_dict, 1, stopwords=guidelines_sw))
        lfs.append(DictionaryLabelingFunction('LF_autoner', autoner_disease_dict, 1, stopwords=guidelines_sw))

        # =======================================================================
        #
        # Custom LFs (Regexes, Dictionaries, Heuristics)
        #
        # =======================================================================

        # common disease char-grams
        ch_rgxs = [
            r'''^(psych|necro|nephro|hyper|throm|hypo|acro|hemo)[A-Za-z]+?([rlt]ic)$''',
            r'''^(hepato|hemato|nephro|cardio|neuro|myelo|oto)*toxic(ities|ity)*$'''
        ]
        lfs.append(RegexEachLabelingFunction('LF_bio_ch_grams_rgx', ch_rgxs, 1))

        # preposition phrases (cancer of the liver)
        anatomy = [
            'central nervous system', 'oral cavity', 'bladder',
            'ureter', 'artery', 'liver', 'aorta', 'brain', 'lung'
        ]
        finding = [
            'adenocarcinoma', 'calcification', 'cancer', 'angiosarcoma',
            'enlargement', 'cirrhosis', 'disorders', 'carcinoma', 'cancer',
            'injury'
        ]
        prep_rgxs = [
            r'''([A-Za-z]+) and ([A-Za-z]+) (insufficiency|dysfunction|(carcinoma|cancer|syndrome|disorder|disease)[s]*)''',
            r'''({}) (in|of) the ({})'''.format(
                '|'.join(finding),
                '|'.join(anatomy)
            )
        ]
        lfs.append(RegexLabelingFunction('LF_prepositions_rgx', prep_rgxs, 1))

        # common slot-filled patterns
        lfs.append(RegexLabelingFunction('LF_common_rgx', [
            r"""\b[A-Za-z-]+'s (syndrome|disease)\b""",
            r'''((artery )*calcification)|(calcification of the [A-Za-z]+)'''], 1))

        # Findings (these findings are considered disorders)
        findings_dict = {
            'hyperhidrosis', 'weight gain', 'loss of consciousness',
            'hypertensive', 'hypotension', 'cardiomegaly', 'hoarding', 'tachyarrhythmias',
            'ventricular tachyarrhythmias', 'weight loss', 'glucosuria'
        }
        lfs.append(DictionaryLabelingFunction('LF_findings', findings_dict, 1))

        # injury / damage
        anat_dict = {
            'renal','myocardial','axonal','neuronal','kidney','bladder',
            'cardiac','hippocampal', 'mitochondrial','cord','hepatic',
            'cerebellum','hepatic','tissue','liver','hepatocellular',
            'myocardial( cell)*','proteinuric','axonal','tissue'}
        regexes = [
            r'''({}) (injury|damage)'''.format('|'.join(sorted(anat_dict,key=len, reverse=1))),
            r'''(malignant ([A-Za-z]+ )*(glioma|tumor)[s]*)''',
            r'''(([A-Za-z]+)'s|wolff[- ]+parkinson[- ]+white|haemolytic[- ]+uraemic|guillain[- ]+barr|hematologic|cholestatic|rabbit)([- ]+like)* syndrome''',
            r'''diabetic( hyperalgesia)*'''
        ]
        lfs.append(RegexLabelingFunction('LF_injuries_rgx', [re.compile(rgx, re.I) for rgx in regexes], 1))

        # Findings dictionary
        lfs.append(OntologyLabelingFunction(f'LF_findings',
                                            findings_ontology,
                                            stopwords=guidelines_sw))

        print(f'Labeling Functions n={len(lfs)}')
        return lfs