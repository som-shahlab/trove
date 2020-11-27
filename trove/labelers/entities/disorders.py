import re
import functools
import itertools
import collections
import pandas as pd

from trove.labelers.tools import *
from trove.labelers.labeling import *
from trove.utils import score_umls_ontologies
from trove.labelers.norm import lowercase, strip_affixes
from trove.labelers.abbreviations import AbbrvDefsLabelingFunction
from trove.labelers.umls import umls_ontology_dicts, load_sem_groups

from trove.labelers.taggers import timex

###############################################################################
#
# Disorders (EHR ShARe/CLEF 2014)
#
###############################################################################

# Single character disorders
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


def LF_single_char_rgx_v1(s):
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

    def lfs(self, train_sentences, top_k=5, active_tiers=None):
        """

        Parameters
        ----------
        train_sentences     unlabeled sentences for ranking ontology coverage
        top_k               use top_k ranked ontologies by coverage + rest

        Returns
        -------

        """
        active_tiers = (1, 2, 3, 4) if active_tiers is None else active_tiers

        sw = set(
            open(f'{self.data_root}/stopwords.txt', 'r').read().splitlines()
        )
        sw = sw.union(set([t[0].upper() + t[1:] for t in sw]))

        # Unified Medical Language System
        umls = load_umls(f'{self.data_root}/ontologies/umls2018AA.parquet', sw)

        print('(Source Vocabulary/Semantic Type) Pairs', len(umls.dictionary))
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


        #Vanderbilt / CARD abbreviations
        cui2sty = pd.read_csv(
            f'{self.data_root}/ontologies/CARD/cui2sty.tsv', sep='\t',
            names=['CUI', 'STY'])
        cui2sty.STY = [sty.replace(" ", "_").lower() for sty in cui2sty.STY]
        cui2sty = {d.CUI: d.STY for d in cui2sty.itertuples()}

        filelist = [
            f'{self.data_root}ontologies/CARD/VABBR_CV_beta.txt',
            f'{self.data_root}ontologies/CARD/VABBR_DS_beta.txt'
        ]
        vabbr = load_vanderbilt_datasets(filelist, self.class_map, cui2sty)

        # wiki abbreviations
        wiki_abbrvs = load_wiki_med_abbrvs(
            f'{self.data_root}ontologies//wiki-med-abbrvs/'
        )

        f_wiki_1 = {}
        f_wiki_2 = {}
        for name in wiki_abbrvs:
            synset = wiki_abbrvs[name]
            for t in wiki_abbrvs[name]:
                if name in class_dictionaries[1]:
                    f_wiki_1[name] = synset
                elif name in class_dictionaries[2]:
                    f_wiki_2[name] = synset

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

        # # OHDSI MedTagger + Stanford SHC COVID-19 terms
        # dict_covid = open(
        #     f'{self.data_root}/dictionaries/covid19.tsv', 'r'
        # ).read().splitlines()
        #


        # ---------------------------------------------------------------------
        # COVID-19 terms
        # ---------------------------------------------------------------------
        filelist = [
            f'{self.data_root}/dictionaries/emerse_covid_19_term_subset.tsv',
            f'{self.data_root}/dictionaries/covid19.tsv'
        ]

        covid_dict = {
            'NOVEL CORONAVIRUS 2019',
            'novel coronavirus 2019',
            'INFLUENZA A/B',
            'influenza a/b',
            'SARS-COV-2',
            'sars-cov-2',
            'RSV',
            'covid-19 virus',
            'COVID', 'COVID-19', 'SARS-COV-2', 'SARS-CoV-2',
            'sars-coronavirus-2', 'covid', 'Covid', 'Covid 19',
            'NOVEL CORONAVIRUS 2019', 'nocov'
        }

        for fpath in filelist:
            with open(fpath,'r') as fp:
                for line in fp:
                    term = line.strip().split("\t")[0]
                    covid_dict.add(term)
                    covid_dict.add(term.lower())

        # ---------------------------------------------------------------------
        # Initialize Ontologies
        # ---------------------------------------------------------------------
        if top_k:
            # rank source vocabs by term coverage
            scores = score_umls_ontologies(train_sentences, umls.dictionary)
            rankings = sorted(scores.items(), key=lambda x: x[-1], reverse=1)
            print(f'Matched {len(rankings)} source vocabularies')
            print(f'top_k={top_k} ({top_k / len(rankings) * 100:2.1f}%)')
            sabs, _ = zip(*rankings[0:top_k])
            sabs = list(sabs) + ['OTHER']
        else:
            # merge all vocabs into a single dictionary
            print('Merge all UMLS source vocabularies')
            sabs = {}

        ontologies = umls_ontology_dicts(sabs, self.class_map, umls)

        if not sabs:
            sabs = ontologies.keys()

        if 4 in active_tiers:
            # Fix special cases
            for name in sorted(ontologies):
                for term in special_cases_dict:
                    if term in ontologies[name]:
                        ontologies[name][term] = [1., 0.]

            # add concepts to ontologies
            negative_phrases = [
                'acute pain service', 'motor vehicle collision',
                'suffers from', 'trauma intensive care unit',
                'list of injuries', 'infectious disease consultation'
            ]
            for term in negative_phrases:
                for name in sorted(ontologies):
                    ontologies[name][term] = [0., 1.]

        # ---------------------------------------------------------------------
        # Labeling Functions
        # ---------------------------------------------------------------------
        lfs = []

        if 4 in active_tiers:
            # no stopwords originally provided by the guidelines
            guidelines_sw = {
                'allergy': 2, 'allergies': 2, 'illness': 2, 'disease': 2,
                'syndrome': 2, 'drug allergies': 2, 'significant': 2, 'cavity': 2,
                'depressed': 2, 'drug': 2, 'right': 2, 'left': 2, 'bilateral': 2,
                'severe': 2, 'painful': 2, 'findings': 2, 'mass': 0, 'OCC': 2,
                'HS': 2, 'AT': 2, 'AS': 2, 'HPI': 2, 'NPH': 2, 'ND': 2, 'RA': 2,
                'OCC': 2, 'NS': 2, 'NT': 2, 'ventricular': 0,

                'CMS-HCC':2, 'CMS':2, 'HCC':2
            }

            guidelines_sw_2 = {
                'allergy': 2, 'allergies': 2, 'illness': 2, 'disease': 2,
                'syndrome': 2, 'drug allergies': 2, 'significant': 2,
                'cavity': 2, 'depressed': 2, 'drug': 2, 'severe': 2, 'painful': 2,
                'findings': 2, 'OCC': 2, 'HS': 2, 'AT': 2, 'AS': 2, 'HPI': 2,
                'NPH': 2, 'ND': 2, 'RA': 2, 'OCC': 2, 'NS': 2, 'NT': 2
            }
        else:
            guidelines_sw = {}
            guidelines_sw_2 = {}

        # guidelines https://drive.google.com/file/d/0B7oJZ-fwZvH5VmhyY3lHRFJhWkk/edit
        fpath = f'{self.data_root}/dictionaries/guidelines/clef-guidelines-disorders.txt'
        guidelines_dict = open(fpath, 'r').read().splitlines()

        # =====================================================================
        #
        # Baseline LFs
        #
        # =====================================================================
        if 1 in active_tiers:
            lfs.append(DictionaryLabelingFunction('LF_stopwords', sw, 2, case_sensitive=True))
            lfs.append(DictionaryLabelingFunction('LF_punct', set('!"#$%&*+,./:;<=>?@[\\]^_`{|}~'), 2))
            lfs.append(RegexEachLabelingFunction('LF_numbers_rgx', [r'''^[-]*[1-9]+[0-9]*([.][0-9]+)*$'''], 2))

            #rgx_punct = [re.compile(r'''\s+[1-9]+[0-9]*([.][0-9]+)*\s+''', re.I)]
            #lfs.append(RegexLabelingFunction('LF_numbers_rgx', rgx_punct, 1))

            lfs.append(DictionaryLabelingFunction('LF_guidelines_1', guidelines_dict, 1))
            lfs.append(DictionaryLabelingFunction('LF_guidelines_2', guidelines_sw_2, 2))

        # =====================================================================
        #
        # Ontology LFs
        #
        # =====================================================================
        if 2 in active_tiers:
            for name in sorted(ontologies):
            #for name in sabs:
                lfs.append(OntologyLabelingFunction(f'LF_{name}',
                                                    ontologies[name],
                                                    stopwords=guidelines_sw))

            # Schwartz-Hearst abbreviations & Synonym sets
            lfs.append(AbbrvDefsLabelingFunction('LF_schwartz_hearst_abbrvs_1', class_dictionaries[1], 1))
            lfs.append(AbbrvDefsLabelingFunction('LF_schwartz_hearst_abbrvs_2', class_dictionaries[2], 2))

        # =====================================================================
        #
        # External Ontologies
        #
        # =====================================================================
        if 3 in active_tiers:
            lfs.append(TermMapLabelingFunction('LF_specialist_synset_1', specialist_1, 1))
            lfs.append(TermMapLabelingFunction('LF_specialist_synset_2', specialist_2, 2))
            lfs.append(TermMapLabelingFunction('LF_adam_synset_1', adam_synset_1, 1))
            lfs.append(TermMapLabelingFunction('LF_adam_synset_2', adam_synset_2, 2))

            lfs.append(TermMapLabelingFunction('LF_vanderbilt_abbrvs_1', vabbr[1], 1))
            lfs.append(TermMapLabelingFunction('LF_vanderbilt_abbrvs_2', vabbr[2], 2))

            lfs.append(DictionaryLabelingFunction('LF_ctd', ctd_disease_dict, 1, stopwords=guidelines_sw))
            lfs.append(DictionaryLabelingFunction('LF_doid', doid_disease_dict, 1, stopwords=guidelines_sw))
            lfs.append(DictionaryLabelingFunction('LF_hp', hp_disease_dict, 1, stopwords=guidelines_sw))
            lfs.append(DictionaryLabelingFunction('LF_autoner', autoner_disease_dict, 1, stopwords=guidelines_sw))

            # COVID-19 terms
            #lfs.append(DictionaryLabelingFunction('LF_covid19', covid_dict, 1,
            #                                      stopwords=guidelines_sw))

        # =====================================================================
        #
        # Custom LFs (Regexes, Dictionaries, Heuristics)
        #
        # =====================================================================
        if 4 in active_tiers:

            # better way to handle numbers
            rgx_numerics = [
                r'''[1-9][0-9]{0,2}([.][0-9]+)*\s*(°|degree[s]*)(f|fahrenheit|c|celcius)''',
                r'''([1-9][0-9]{0,2}|[0][.][0-9]+)([%]|percent)''',
                r'''[1-9][,]*[0-9]{0,3}\s*(mg|ml|mcg/hr|mcg|gram)''',
                r'''[(]*[1-9][0-9]{2}[)]*[-][0-9]{3}[-][0-9]{4}|(pgr|pager)\s*[0-9]{3,}''',
                # HACK for long sequences of punctuation
                r'''[*=_-]{3,}'''
            ]

            rgx_numerics = [re.compile(rgx, re.I) for rgx in rgx_numerics]
            lfs.append(RegexLabelingFunction('LF_numeric_concepts_rgx_2', rgx_numerics, 2))

            # TIMEX3 mentions
            lfs.append(RegexLabelingFunction('LF_timex_2', timex.regexes, 2))

            # ---------------------------------------------------------------------
            # Findings dictionary
            # ---------------------------------------------------------------------
            lfs.append(OntologyLabelingFunction(f'LF_findings', findings_ontology,
                                                stopwords=guidelines_sw))

            #------------------------------------------------------------------
            # Misc disorders
            #------------------------------------------------------------------
            rgx_disorders = [
                # masses
                r'''(non[-]*)*calcified (nodule|granuloma|plaquing|mass[e]*)[s]*''',
                r'''(((epi|calci)[a-z]+|lung|hilar|renal|sacral|pancreas|ampullary|bladder|underlying|LV) )*mass(es)*''',
                r'''\b(mass(es)*)\b''',
                # respiratory
                r'''(bundle-branch block|pleural effusion[s]*)''',
                r'''(decreased|mechanical|coarse) (upper airway|breath|bowel|heart) sound[s]*''',
                # fractures
                r'''(((radial|superior|proximal|healed|nondisplaced|open) )*((transverse process|(pubic )*rami|iliac bone|fibula|ulnar|sacral|pelvic|neck|rami|C2) )*)*fracture''',
                # heart-related
                r'''(wide complex )*tachy(cardi[a]*c|pneic)*''',
                r'''((small|multi|four|three|two|one)\s*[-]\s*vessel|[1234]\s*[-]\s*vessel|coronary artery|atherosclerotic|atheromatous|LM(CA)*) disease''',
                r'''((left main|right|left)\s*coronary artery|LCMA|RCA) (disease|stenosis)''',
                r'''((impaired|poor|depressed|right|left)\s*)(ventricular) (ejection fraction|hypertrophy|function|abnormalities)''',
                # abnormalities
                r'''(abnormal(ities|ity)*) (PAP smear|septal motion|movement|alignment|period|signal|EEG)[s]*''',
                r'''((ST((or|[- ])T)*\s*)|T)( wave\s*)*(changes|abnormalities)''',
                # misc
                r'''food (impaction[s]*|impacted)''',
                r'''((chronic|sinus)\s*)*(mucosal )(thickening|irregularity|changes)''',
                # bleeding
                r'''((heavy\s*)\s*)*((brain metastatic|vaginal|gastrointestinal|hypertensive|gastric ulcer|postpartum|intraoperative|intra[- ]*cranial|gi)\s*)*(bleed(ing)*)''',
                r'''((non[- ]*bleeding) ((esophageal)\s)*(varices|ulcer))''',
                r'''(bleeding (ulcer|gastric ulcer|problem[s]*))''',
                # laceration / lesion
                r'''((skin|sclerotic|gastric|ostial|sacral|liver|deformity|facial|splenic)\s*)*((laceration|lesion)[s]*)'''
            ]

            # ventricular function
            prefix = r'''(left|right|nonsustained|hyperdynamic|impaired|poor|slow|first[- ]+degree)'''
            anat = r'''(non-sinus|(supra|intra|atrio)*ventricular|wall|end)'''
            root = r'''(rhythm|end diastolic pressure|blood product[s]*|motion abnormalities|outflow tract obstruction|enlargement|block|response|systolic dysfunction|septal defect|hypertrophy|inflow pattern|fibrillation|dilatation|hemorrhage|diastolic collapse|failure|dysfunction|tachycardia arrest|tachycardia[s]*|arrythmia[s]*|ectopy|conduction delay)'''
            rgx_disorders += [f'''({prefix}\s*)*({anat}\s*)+{root}''']

            # pneumonia
            prefix = r'''(RUL|((right|left)( middle)* lobe)|aspiration|mycoplasma|pneumocystis carinii|klebsiella|ventilator[- ]related)'''
            rgx_disorders +=  [f'''({prefix}\s*)*pneumonia''']

            rgx_disorders = [re.compile(rgx, re.I) for rgx in rgx_disorders]
            lfs.append(RegexLabelingFunction('LF_disorders_rgx_1', rgx_disorders, 1))

            #------------------------------------------------------------------
            # Changes
            #------------------------------------------------------------------
            rgx_changes = [
                r'''(mental status|(nonspecific )*(ST(-T)*|T)( (segment|wave))*|EKG|MS|vision|cardi[ao]pulm(onary)*|weight) change[s]*'''
            ]
            lfs.append(RegexLabelingFunction('LF_changes_rgx_1', rgx_changes, 1))

            # -----------------------------------------------------------------
            # Units / Roles
            # -----------------------------------------------------------------
            depts = {
                'neuro[-]*oncology', 'neurosurgery',
                'radiation oncology', 'trauma', 'oncology',
                '(ct )*surgery', 'infectious disease',
                'intensive care', 'operating', 'emergency'
            }
            healthcare_rgxs = [
                f'''({"|".join(depts)}) ((was|were) consulted|consult|consulted|consultation)''',
                f'''({"|".join(depts)}) (service|unit|room|bay|dept|department)''',
                r'''psychiatric (nurse|case manager|stay|facility)|mental health provider'''
            ]
            lfs.append(RegexLabelingFunction('LF_units_2', healthcare_rgxs, 2))

            #------------------------------------------------------------------
            # Pulmonary disease & spatial
            #------------------------------------------------------------------
            rgx_pulmonary = [
                r'''(chronic obstructive pulmonary disease|((chronic|acute|ischemic) )*(cardio)*pulmonary (artery hypertension|vascular congestion|hyperinflation|consolidation|hypertension|infiltrate|sarcoidosis|contusion|process|effusion|disease|overload|embolism|sequelae|process|embolus|nodule|status|edema)[s]*)''',
            ]
            rgx_spatial = [
                r'''((right|left|L|R)[- ]+side[d]*|bilateral|right/left|right|left|lower|upper|R[/ ]+L|RUL|RUQ|LE) (ventricular enlargement|gastrointestinal bleed|gastrointestinal bleed|abdominal tenderness|atrial enlargement|extremity edema|disorientation|septic emboli|pneumothorax|hemiparesis|hemiparesis|infiltrates|tenderness|pneumonia|confusion|effusions|confusion|weakness|hematoma|edema)''',
            ]

            rgx_pulmonary = [re.compile(rgx, re.I) for rgx in rgx_pulmonary] + rgx_spatial
            lfs.append(RegexLabelingFunction('LF_pulmonary_spatial_rgx_1', rgx_pulmonary, 1))

            #------------------------------------------------------------------
            # Misc. custom abbreviations
            #------------------------------------------------------------------
            non_disorder_abbrvs_dict = {
                'MVC', 'HD', 'BMD', 'CCA', 'SEM', 'PICA', 'AKU', 'CMO', 'AT',
                'INR', 'LAD', 'ACLS', 'WED', 'SDA', 'ABD', 'NPH', 'EXT', 'FBS',
                'LCA', 'ID', 'FHL', 'OCC', 'HS', 'AT', 'AS', 'HPI', 'NPH', 'EXT',
                'RESP', 'NPH', 'MON', 'TUE', 'WED', 'THUR', 'FRI', 'SAT', 'SUN'
            }
            disorder_abbrvs_dict = {
                'AH', 'APD', 'CAD', 'CKD', 'CMV', 'CP', 'CPP', 'DOE', 'GCS', 'GIB',
                'GSW', 'LBP', 'LOC', 'LVH', 'NS', 'NSR', 'OD', 'OSA', 'PNA', 'SAH',
                'SBP', 'SOB', 'UTI', 'PNA', 'LVOT', 'LVH', 'DNI', 'INR'
            }

            custom_abbrvs = {}
            for t in disorder_abbrvs_dict:
                custom_abbrvs[t] = [1.,0.]
            for t in non_disorder_abbrvs_dict:
                custom_abbrvs[t] = [0.,1.]

            lfs.append(OntologyLabelingFunction(f'LF_custom_abbrvs', custom_abbrvs, stopwords = guidelines_sw))

            #------------------------------------------------------------------
            # UMLS error concepts
            #------------------------------------------------------------------
            # these are UMLS concepts that are not Disorder concepts and t
            # hus are labeled incorrectly with UMLS binary supervision
            umls_err_dict = set([
                'fall', 'bleed', 'lesion', 'lesions', 'swelling', 'akinetic',
                'tamponade', 'complaints', 'unresponsive', 'malalignment',
                'mental status', 'consolidation', 'opacification',
                'acute distress', 'carotid bruits', 'opacities'
            ])
            lfs.append(DictionaryLabelingFunction('LF_umls_non_diso_1', umls_err_dict, 1))

            #------------------------------------------------------------------
            # Single characters (n/v/m)
            #------------------------------------------------------------------
            anchor_lf = [lf for lf in lfs if lf.name == 'LF_pulmonary_spatial_rgx_1']
            if not anchor_lf:
                anchor_lf = [lf for lf in lfs if lf.name == 'LF_OTHER']
            anchor_lf = anchor_lf[0]

            lf_char = functools.partial(LF_single_char_rgx, dict_lf=anchor_lf)
            lfs.append(CustomLabelingFunction("LF_single_char_1", lf_char))

            # ---------------------------------------------------------------------
            # medical shorthand
            # ---------------------------------------------------------------------
            med_shorthand_dict = set([
                'h/o', 's/p', "d/c'd", 'd/c', "d/c'ed", 'd/ced', 'bp', 'y/o',
                'c/b', 'i/p', 'd/t', 'n/a', 'w/o', 'w/d', 'b/l', 'c/w', 'p/w',
                'w/', 'f/u', '+/-', 'u/s', 'o/w',
            ])
            lfs.append(DictionaryLabelingFunction('LF_med_shorthand_2',
                                                  med_shorthand_dict, 2,
                                                  max_ngrams=5))

            # ---------------------------------------------------------------------
            # masses and ST-T wave changes
            # ---------------------------------------------------------------------
            misc_rgx = [
                r'''((ST((or|[- ])T)*\s*)|T)( wave\s*)*(changes|abnormalities)''',
                r'''\b(mass(es))\b'''
            ]
            lfs.append(
                UnipolarUnionLabelingFunction('LF_masses_1', [
                    RegexLabelingFunction('LF_masses', misc_rgx, 1), anchor_lf], 1)
            )

        print(f'Labeling Functions n={len(lfs)}')
        return lfs
