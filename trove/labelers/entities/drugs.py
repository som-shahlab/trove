import re
import functools
import itertools
import collections
import pandas as pd
from functools import partial
from trove.labelers.tools import *
from trove.labelers.labeling import *
from trove.utils import score_umls_ontologies
from trove.labelers.norm import lowercase, strip_affixes
from trove.labelers.abbreviations import AbbrvDefsLabelingFunction
from trove.labelers.umls import umls_ontology_dicts, load_sem_groups
from trove.labelers.spans.negex import NegEx
from trove.labelers.spans.taggers import (
    get_left_span,
    token_distance,
    match_regex,
    get_between_span
)
from trove.labelers.taggers import timex

###############################################################################
#
# i2b2 Drug 2009
#
###############################################################################

def negex_definite_left(span, negex):
    text = get_left_span(span, span.sentence, window=2).text
    rgx = negex.rgxs['definite']['left']
    return True if rgx.search(text) else False

#
# def invert_label(span, header_rgx, negation_func):
#     """
#     If mention is any one of these cases, it is not labeled as per i2b2 guidelines:
#     - The medication was not actually perscribed or taken by the patient
#       e.g., negation, family member meds, headers.
#     - The medication is listed as an allergy
#     """
#     regexes = [
#         r'''\b(allergies[:]*|allergy|allergic to|allergic reaction)\b''',
#         r'''\b(((grand)*(mother|father)|grand(m|p)a)([']*s)*|((parent|(daught|sist|broth)er|son|cousin)([']*s)*))\b'''
#     ]
#     # reject headers
#     if re.search(header_rgx, span.sentence.text):
#         return True
#
#     # reject allergies
#     left = get_left_span(span)
#     for rgx in regexes:
#         match = match_regex(rgx, left)
#         if match:
#             d = token_distance(match, span)
#             if d <= 50:
#                 return True
#
#     # reject negated entities
#     if negation_func(span):
#         return True
#     return False


def invert_label(span, header_rgx, negation_func):
    """
    If mention is any one of these cases, it is not labeled as per i2b2 guidelines:
    - The medication was not actually perscribed or taken by the patient
      e.g., negation, family member meds, headers.
    - The medication is listed as an allergy
    """

    regexes = [
        r'''(ALLERGIES [:])''',
        r'''\b(allergies\s*[:]*|allergic (to|reaction)|(allergy|sensitive) to|allergy)\b''',
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

    # skip pseudo negation
    neg = r'''\b(no relief with)\b'''
    match = match_regex(neg, left)
    if match:
        d = token_distance(match, span)
        btw = get_between_span(match, span)
        if d <= 3 and (not btw or not re.search(r'''[,.;]''', btw.text)):
            return False

    # reject negated entities
    if negation_func(span):
        return True

    neg = r'''\b((neg|negative) for|not (been )*started|no)\b'''
    match = match_regex(neg, left)
    if match:
        d = token_distance(match, span)
        btw = get_between_span(match, span)
        if d <= 3 and (not btw or not re.search(r'''[,.;]''', btw.text)):
            return True

    # reject historical entities
    hist = r'''\b(history of)\b'''
    match = match_regex(hist, left)
    if match:
        d = token_distance(match, span)
        btw = get_between_span(match, span)
        if d <= 3 and (not btw or not re.search(r'''[,.;]''', btw.text)):
            return True

    return False


class i2b2DrugLabelingFunctions(object):
    """
    The 2009 i2b2 medication dataset combines multiple layers of annotation
    logic, which poses complexities in labeling function design for
    dictionary-based weak supervision. Guidelines dictate:
    - Only label medications the patient is currently taking
    - Do not label medication allergies

    We use a more complex templated labeling function that allows us to remap
    span labels using another heuristic.

    """
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
            'receptor',
            'indicator,_reagent,_or_diagnostic_aid',
            'biomedical_or_dental_material',
            'clinical_attribute',
            'enzyme',
            'immunologic_factor',
            'organic_chemical',
            'amino_acid,_peptide,_or_protein',
            'gene_or_genome',
            'amino_acid_sequence',
            'nucleotide_sequence',
            'nucleic_acid,_nucleoside,_or_nucleotide'
        ]

        for sty in ignore:
            if sty in class_map:
                del class_map[sty]

        rm = [sty for sty in class_map if class_map[sty] == 0]
        for sty in rm:
            del class_map[sty]

        return class_map


    def lfs(self, train_sentences, top_k=10, active_tiers=None):

        active_tiers = (1, 2, 3, 4) if active_tiers is None else active_tiers

        # ---------------------------------------------------------------------
        # Supervision Sources
        # ---------------------------------------------------------------------
        print('Loading Ontologies/Dictionaries')
        print('=' * 80)
        sw = set(
            open(f'{self.data_root}/stopwords.txt', 'r').read().splitlines()
        )
        sw = sw.union(set([t[0].upper() + t[1:] for t in sw]))

        # UMLS
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

        print(ontologies.keys())
        print(sabs)
        # ---------------------------------------------------------------------
        # Labeling Functions
        # ---------------------------------------------------------------------

        reject_headers = {
            'SOCIAL HISTORY', 'ALLERGY', 'ALLERGIES', 'CC', 'DIET',
             'NEUROLOGICAL', 'LABORATORY DATA', 'CHIEF COMPLAINT',
             'FAMILY HISTORY', 'EKG', 'NEURO', 'PAST SURGICAL HISTORY',
             'FAMILY HISTORY', 'ALLG', 'LABORATORY', 'PREOPERATIVE LABS',
             'LABS'}
        header_rgx = r'''^({})[:]'''.format(
            '|'.join(sorted(reject_headers, key=len, reverse=1))
        )

        negex = NegEx(data_root=f'{self.data_root}/negex/')
        is_negated = partial(negex_definite_left, negex=negex)
        override_span_label = partial(invert_label,
                                      header_rgx=header_rgx,
                                      negation_func=is_negated)

        guidelines_sw = {'tobacco', 'alcohol', 'water', 'blood pressure'}

        # custom stopwords
        # These prevent biases from the main ontologies using the independent snorkel model.
        # For example, "blood pressure medications", blood pressure is a negative concept in all UMLS ontologies
        # and we don't capture the chain dep linking it to medications by default.
        guidelines_sw = guidelines_sw.union({
            'NAD', 'nad', 'fat', 'sugar', 'PCP', 'pcp',
            'chemo', 'chemotherapy', 'lipase', 'amylase',
            'smoke', 'mg', 'NPH', 'nph', 'INH',
            'calcium', 'magnesium', 'iron', 'potassium',
            'glucose', 'phosphate', 'electrolyte', 'salt',
            'meds', 'medications', 'medication', 'drug', 'gentle',
            'tomorrow', 'dose', 'unknown', 'air', 'toxin', 'prevent',
            'increase', 'subcutaneous', 'intravenous', 'diurese',
            'pain', 'blood pressure', 'oral', 'coated', 'enteric',
            'topical', 'immediate', 'release', 'slow', 'mg',
            'mercury', 'basis', 'FLAIR', 'products', 'hypoglycemic',
            'blood', 'drip', 'at', 'each', 'daily',
            'red blood cells', 'immediate', 'release',
            'lead', 'leads', 'nitrites', 'nitrite', 'stent', 'milligrams',
            'MIBI', 'mibi', 'Mibi',  'adenosine', 'Adenosine MIBI',
            'subcutaneous', 'unfractionated', 'intravenous', 'O2',
             })

        # Common lab values (i.e., not drugs)
        labs = {
            'hemoglobin', 'sodium', 'bicarbonate', 'phosphorus', 'magnesium',
            'lactate', 'PCO2', 'white count', 'lipase',
            'platelets', 'alk phos', 'white blood cell count', 'phosphate',
            'MB', 'PT', 'serum osms', 'AST', 'amylase',
            'CO2', 'potassium', 'carbon dioxide', 'creatinine', 'glucose',
            'troponin', 'PTT', 'white blood cell[s]*',
            'BUN', 'calcium', 'WBC', 'chloride', 'bicarb', 'ALT',
            'O2 saturation', 'beta hydroxybutyrate', 'anion gap',
            'TSH', 'lipase', 'PO2', 'platelet count', 'CK', 'hematocrit',
            'BNP', 'ABG', 'T[.]* bili', 'INR', 'ABG T',
            '[TD]-bili', 'T-bili', 'D-bili', 'o2 sat', 'o2 staturation', 'hCG',
            'O2 sat', 'O2 SAT', 'o2 saturation',
            'T3', 'T4', 'acetone', 'coagulation factors', 'lactic acid', 'PTH',
            'blood sugar'
        }

        guidelines_sw = guidelines_sw.union(set(labs))
        guidelines_sw = guidelines_sw.union({t + 's' for t in guidelines_sw})

        if 4 not in active_tiers:
            # don't use any custom stopwords
            guidelines_sw = {}

        MAX_NGRAMS = 8

        # Preliminary.Annotation.Guidelines.6.12.pdf
        fpath = f'{self.data_root}/dictionaries/guidelines/i2b2-guidelines-drugs.txt'
        guidelines_dict = open(fpath,'r').read().splitlines()
        guidelines_dict = set(guidelines_dict).union({t.lower() for t in guidelines_dict})

        lfs = []

        # =======================================================================
        #
        # Baseline LFs
        #
        # =======================================================================
        if 1 in active_tiers:
            lfs.append(DictionaryLabelingFunction('LF_stopwords', sw, 2, case_sensitive=True))
            #lfs.append(DictionaryLabelingFunction('LF_punct', set('!"#$%&*+,./:;<=>?@[\\]^_`{|}~'), 2))

            lf1 = DictionaryLabelingFunction('LF_punct', set('!"#$%&*+,./:;<=>?@[\\]^_`{|}~'), 2)
            lf2 = RegexLabelingFunction('LF_punct_2', [re.compile(r'''([&][A-Za-z]+[;]|[*_-]{2,})''', re.I)], 2)
            lfs.append(UnipolarUnionLabelingFunction('LF_punct', [lf1, lf2]))

            lfs.append(RegexEachLabelingFunction('LF_numbers_rgx', [r'''^[-]*[1-9]+[0-9]*([.][0-9]+)*$'''], 2))
            lfs.append(DictionaryLabelingFunction('LF_guidelines', guidelines_dict, 1))
            #lfs.append(RegexEachLabelingFunction('LF_numbers_rgx', [r'''^[-]*[1-9]+[0-9]*([.][0-9]+)*$'''], 2))
            #lfs.append(DictionaryLabelingFunction('LF_guidelines', guidelines_dict, 1))

        # =======================================================================
        #
        # UMLS Ontology LFs
        #
        # =======================================================================
        if 2 in active_tiers:

            if 4 not in active_tiers:
                # Guideline only patterns
                slot_rgxs = [
                    '{} (nebulizer|solution|cream|tablet)[s]*',
                    '{} [(]\s*{}\s*[)]',
                    '{} [(]\s*{} (and|[/-]) {}\s*[)]',
                    '{} [(]\s*([A-Za-z0-9-\\\]+\s*){{1,3}}\s*[)]',
                    '[A-Za-z0-9-]+ [(]\s*{}\s*[)]',
                    '{}\s*[)]',
                    '[(]\s*{}'
                ]
                span_rule = None
            else:
                # expanded patterns
                slot_rgxs = [
                    '{} (nebulizer|ointment|inhaler|solution|cream|paste|tablet|syrup|(nitro)*spray|patch|elixer|neb|(([1-9][%] )*(topical )*)*(powder|cream|patch))[s]*',
                    '{} (extended|sust[.]|sustained|immediate|immed[.]*|slow) (release|releas|rel[.]*)\s*[)]*',
                    '{} (enteric coated|enteric\s*[-]\s*coated)',
                    '{} [(]\s*{}\s*[)]',
                    '{} [(]\s*{} (and|[/-]) {}\s*[)]',
                    '{} [(]\s*([A-Za-z0-9-\\\]+\s*){{1,3}}\s*[)]',
                    '[A-Za-z0-9-]+ [(]\s*{}\s*[)]',
                    '{}\s*[)]',
                    '[(]\s*{}'
                ]
                # negation, allergies, non-active medications
                span_rule = override_span_label

            for name in sorted(ontologies):
            #for name in sabs:
                lf = SlotFilledOntologyLabelingFunction(f'LF_{name}',
                                                        ontologies[name],
                                                        max_ngrams=MAX_NGRAMS,
                                                        slot_patterns=slot_rgxs,
                                                        stopwords=guidelines_sw,
                                                        span_rule=span_rule)
                lfs.append(lf)

            # Schwartz-Hearst abbreviations
            lfs.append(AbbrvDefsLabelingFunction('LF_schwartz_hearst_abbrvs_1', class_dictionaries[1], 1))
            lfs.append(AbbrvDefsLabelingFunction('LF_schwartz_hearst_abbrvs_2', class_dictionaries[2], 2))

        # =======================================================================
        #
        # External Ontologies
        #
        # =======================================================================
        if 3 in active_tiers:
            lfs.append(TermMapLabelingFunction('LF_specialist_synset_1', specialist_1, 1))
            lfs.append(TermMapLabelingFunction('LF_specialist_synset_2', specialist_2, 2))
            lfs.append(TermMapLabelingFunction('LF_adam_synset_1', adam_synset_1, 1))
            lfs.append(TermMapLabelingFunction('LF_adam_synset_2', adam_synset_2, 2))

            lfs.append(DictionaryLabelingFunction('LF_ctd', ctd_chemical_dict, 1, max_ngrams=MAX_NGRAMS, stopwords=guidelines_sw))
            lfs.append(DictionaryLabelingFunction('LF_doid', doid_disease_dict, 2, max_ngrams=MAX_NGRAMS, stopwords=guidelines_sw))
            lfs.append(DictionaryLabelingFunction('LF_hp', hp_disease_dict, 2, max_ngrams=MAX_NGRAMS, stopwords=guidelines_sw))


        # =======================================================================
        #
        # Custom LFs (Regexes, Dictionaries, Heuristics)
        #
        # =======================================================================
        if 4 in active_tiers:
            # Custom LFs for this task are more complicated due to i2b2's drug
            # definitions, which includes somewhat ad hoc inclusion criteria
            # of modes, as well as *not* annotating negated, allergy mentions

            # -----------------------------------------------------------------
            # Illicit drugs
            # -----------------------------------------------------------------
            illicit_drugs_dict = open(f'{self.data_root}/dictionaries/illicit_drugs.txt','r').read().splitlines()
            lfs.append(DictionaryLabelingFunction('LF_illicit_drugs', illicit_drugs_dict, 2, case_sensitive=True))

            # -----------------------------------------------------------------
            # Over-the-counter drugs
            # -----------------------------------------------------------------
            rgx_otc = re.compile(r'''\b(mouth wash|eye drop|toe cream|stool softener)[es]*\b''', re.I)
            lfs.append(RegexLabelingFunction('LF_otc_rgx', [rgx_otc], 1))

            # -----------------------------------------------------------------
            # Insulin
            # -----------------------------------------------------------------
            rgx_insulin_pos = [
                r'''(lantus [(] insulin glargine [)]|NPH\s*(with)*insulin|insulin (NPH|regular)*\s*(aspart|human)|insulin(\s+(NPH|lispro|regular|regimen|drip|lente|[0-9]+/[0-9]+))*)''',
                r'''(((NPH )*humulin|regular|human|lantus|NPH) )*insulin''',
                r'''(insulin [0-9]+/[0-9]+( human)*)''',
                r'''(insulin\s*NPH|NPH(\s*humulin)*\s*insulin|NPH)''',
            ]
            rgx_insulin_pos = [re.compile(rgx, re.I) for rgx in rgx_insulin_pos]
            lfs.append(RegexLabelingFunction('LF_insulin_rgx_1', rgx_insulin_pos, 1))

            # -----------------------------------------------------------------
            # Not insulin mentions
            # -----------------------------------------------------------------
            rgx_insulin_neg = [
                r'''(insulin[- ]dependent)''',
                r'''((non[- ]*)*insulin[- ]dependent([ ,](diabetes mellitus|diabetes|hypothyroidism))*|non[- ]*insulin|insulin sliding scale)'''
            ]
            rgx_insulin_neg = [re.compile(rgx, re.I) for rgx in rgx_insulin_neg]
            lfs.append(RegexLabelingFunction('LF_insulin_rgx_2', rgx_insulin_neg, 2))

            # -----------------------------------------------------------------
            # Misc medications
            # -----------------------------------------------------------------
            rgx_drugs = [
                r'''(CCB|KCL)''',
                r'''(vancomycin|levofloxacin|flagyl|iron supplementation)''',
                r'''(KCL\s*(IMMEDIATE|SLOW)\s*(RELEASE|REL[.]*)|asa)''',
                r'''(KCL|potassium chloride)\s*(immediate|slow)\s*(release|rel[.]*)''',
                r'''(chemotherapy|chemo)( regimen[s]*)*''',
                r'''\b(ACEi|ACE INHIBITORS)\b''',
                r'''(metamucil\s*(sugar free)*)''',
                r'''(prednisolone [0-9]{1,2}[%])''',
                r'''(red blood|red blood cell|rbc|platelet) transfusion[s]*''',
                r'''(b|beta)[-]*blocker[s]*''',
                r'''(cardiac|cardiovascular|diabetes|immunosuppressive|pain|copd|(blood\s*)*pressure|oral|outpatient|home|these|your|pressor|pressure)\s*(med(ication|icine)*[s]*)''',
                r'''(metoprolol xl|kcl immediate release)'''
            ]
            rgx_drugs = [re.compile(rgx, re.I) for rgx in rgx_drugs]
            lfs.append(RegexLabelingFunction('LF_drugs_rgx', rgx_drugs, 1))

            # -----------------------------------------------------------------
            # Vitamins
            # -----------------------------------------------------------------
            vit_rgx_1 = re.compile(r'''(vitamin|vit[.]*)[s]*[- ][abcde][-]*[1-3]*''', re.I)
            lfs.append(RegexLabelingFunction('LF_vitamins_rgx_1', [vit_rgx_1], 1))

            # -----------------------------------------------------------------
            # Negated vitamins
            # -----------------------------------------------------------------
            vit_rgx_2 = re.compile(r'''no ([A-Za-z]+\s*){1,2}(vitamin|vit[.]*)[s]*[- ][abcdek][-]*[1-3]*''', re.I)
            lfs.append(RegexLabelingFunction('LF_vitamins_rgx_2', [vit_rgx_2], 2))

            non_drug_concepts_rgxs = [
                # Section Headers
                header_rgx,
                # Lab values e.g., 'phosphate 0.8'
                r'''((?:{})(?: (?:was|of|at))* [0-9]{{1,3}}(?:[.][0-9]+)*[%]*)'''.format('|'.join(sorted(labs, key=len, reverse=1))),
                r'''[0-9]*\s*(to|[-])\s*[0-9]*\s*(red blood cell[s]*)''',
                # Generic mentions of "medications"
                r'''(medication[s]*)''',
                # Drug/chemical mention but not as a prescribed drug
                r'''(potassium chloride policy|(cardiac|adenosine|guanosine)[- ]*mibi|mibi)''',
                # Oxygen saturation
                r'''(o2|oxygen) (saturation|sat)(s)*(\s*[0-9]+[%]*)*''',
                # Allergies
                r'''((allergy|allergies)\s*[:]\s*([A-Za-z]+[ ,]*)+)''' ,
                r'''(([A-Za-z]+)\s*){1,2}\s*(cause(s|d)*)\s*([A-Za-z]+)*\s*(rash|headache|nausea|swelling|bleeding|reaction)+''',
                # diabetes-related insulin mentions
                r'''((non[- ]*)*insulin[- ]dependent([ ,](diabetes mellitus|diabetes|hypothyroidism))*|non[- ]*insulin|insulin sliding scale)''',
                r'''(diabetes mellitus insulin therapy|insulin toxicity|type 2 diabetes mellitus\s*[,]\s*insulin)''',
                # Proper names
                r'''(coumadin (clinic|program)|oral medicine|captopril renogram test)''',
                # Parenthetical comments
                r'''\( (dose unknown) \)'''
            ]
            non_drug_concepts_rgxs = [re.compile(r,re.I) for r in non_drug_concepts_rgxs]
            lfs.append(RegexLabelingFunction('LF_drug_concepts_rgx_2', non_drug_concepts_rgxs, 2))

            # -----------------------------------------------------------------
            # Drug concepts
            # -----------------------------------------------------------------
            dose = r'''[(] [0][.][1-9] (mg) [)]'''
            nitro = r'''(nitroglycerin\s*(((sublingual )*tablet|patch|spray|paste)[s]*)*|nitro[ ]*(spray|patch|paste)|ntg|nitro)[es]*\s*(1/150|(sublingual )*tablet[s]*|spray[s])*'''
            nitro_rgx = f'''({nitro})\s*({dose})*'''

            drug_concepts_rgxs = [
                r'''(metoprolol\s*(tartrate|sustained release)*|toprol[ -]*(xl)*\s*(\( metoprolol\s*(tartrate)* \))*)''',
                r'''toprol xl \( (metoprolol\s*(succinate extended release|\( sust[.] rel[.] \))) \)''',
                r'''(metoprolol xl|kcl immediate release)''',
                r'''(iron|potassium|folate)\s*(supplement(s|ation)*)''',
                # r'''(steroids|chemotherapy)''',
                r'''(steroids)''',
                r'''(multi|b|maalox|short)[- ](acting|blocker|tablet|vitamin)[s]*''',
                r'''(antibiotic|anti[- ]ischemic|diabetic|torsemide|novolog|medication) regimen''',
                r'''((enteric[- ])*coated\s*(aspirin|asa)|(ecasa|ec asa|enteric coated asa)\s*\( aspirin enteric coated \))''',
                r'''(insulin (7/30|regular|nph) human)''',
                r'''(((cardiac|pain|copd|(blood\s*)*pressure|oral|narcotic pain|outpatient|these|your|pressor|pressure)\s*)(medication[s]*))''',
                r'''((bp|diabetes|sedating|immunosuppressive) (medication|med)[s]*)''',
                nitro_rgx
            ]
            drug_concepts_rgxs = [re.compile(r,re.I) for r in drug_concepts_rgxs]
            lfs.append(RegexLabelingFunction('LF_drug_concepts_rgx_1', drug_concepts_rgxs, 1))

            # -----------------------------------------------------------------
            # Head/tail
            # -----------------------------------------------------------------
            # tail of med mentions
            tail_rgx = r'''(human|nebulizer|liquid|sr|mg|release|rel[.]*|patch|liquid|[.]{3})\s*([)]\s*)+'''
            tail_rgx = re.compile(tail_rgx, re.I)

            # head of med mentions
            head_rgx = r'''\s*(XL|MG|REL[.]*|solution|therapeutic|nebulizer|1/150|70/30)\s*([(]\s*)+'''
            head_rgx = re.compile(head_rgx, re.I)

            # 10x longer eval time due to prefix word
            word_rgx = r'''([A-Za-z0-9/.+=-]+\s*){1,5}'''
            nested_rgx = f'''([A-Za-z0-9/.+=-]+\s*)([0-9/]+\s*)*[(]\s*{word_rgx}\s*[(]\s*{word_rgx}\s*[)]\s*[)]'''
            # nested_rgx = f'''[(]\s*{word_rgx}\s*[(]\s*{word_rgx}\s*[)]\s*[)]'''
            nested_rgx = re.compile(nested_rgx, re.I)

            # slash
            meds_rgx = 'nitrate|lipitor|niacin|calcium|diuril|lasix|plavix|statin|flagyl|ACEi|coum|levo|dig|asa|mag|BB|k'
            slash_rgx = f'''(({meds_rgx})+[/]){{1,3}}({meds_rgx})+'''
            slash_rgx = re.compile(slash_rgx, re.I)

            # punctuation
            punct_rgx = r'''((enteric|ACE)\s*[-]|[.]{3}|((quick )*dissolve|chewable|asa)\s*[/]\s*(chewable)*)'''
            punct_rgx = re.compile(punct_rgx, re.I)

            lfs.append(RegexLabelingFunction('LF_glue_rgx', [nested_rgx, head_rgx, slash_rgx, punct_rgx, tail_rgx], 1))

            # TIMEX3 mentions
            lfs.append(RegexLabelingFunction('LF_timex_2', timex.regexes, 2))


        print(f'Labeling Functions n={len(lfs)}')
        return lfs


###############################################################################
#
# Drug
#
###############################################################################

class DrugLabelingFunctionsOLD(object):
    """
    i2b2 labeling functions w/o allergy and negation logic
    """
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
            'receptor',
            'indicator,_reagent,_or_diagnostic_aid',
            'biomedical_or_dental_material',
            'clinical_attribute',
            'enzyme',
            'immunologic_factor',
            'organic_chemical',
            'amino_acid,_peptide,_or_protein',
            'gene_or_genome',
            'amino_acid_sequence',
            'nucleotide_sequence',
            'nucleic_acid,_nucleoside,_or_nucleotide'
        ]

        for sty in ignore:
            if sty in class_map:
                del class_map[sty]

        rm = [sty for sty in class_map if class_map[sty] == 0]
        for sty in rm:
            del class_map[sty]

        return class_map


    def lfs(self, train_sentences, top_k=10, active_tiers=None):

        # ---------------------------------------------------------------------
        # Supervision Sources
        # ---------------------------------------------------------------------
        print('Loading Ontologies/Dictionaries')
        print('=' * 80)
        sw = set(
            open(f'{self.data_root}/stopwords.txt', 'r').read().splitlines()
        )
        sw = sw.union(set([t[0].upper() + t[1:] for t in sw]))

        # UMLS
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
        if top_k:
            # rank source vocabs by term coverage
            scores = score_umls_ontologies(train_sentences, umls.dictionary)
            rankings = sorted(scores.items(), key=lambda x: x[-1], reverse=1)
            print(f'Matched {len(rankings)} source vocabularies')
            print(f'top_k={top_k} ({top_k / len(rankings) * 100:2.1f}%)')
            sabs, _ = zip(*rankings[0:top_k])
        else:
            # merge all vocabs into a single dictionary
            print('Merge all UMLS source vocabularies')
            sabs = {}

        ontologies = umls_ontology_dicts(sabs, self.class_map, umls)

        # ---------------------------------------------------------------------
        # Labeling Functions
        # ---------------------------------------------------------------------

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
        fpath = f'{self.data_root}/dictionaries/guidelines/i2b2-guidelines-drugs.txt'
        guidelines_dict = open(fpath,'r').read().splitlines()
        guidelines_dict = set(guidelines_dict).union({t.lower() for t in guidelines_dict})

        lfs = []

        # =======================================================================
        #
        # Baseline LFs
        #
        # =======================================================================

        lfs.append(DictionaryLabelingFunction('LF_stopwords', sw, 2, case_sensitive=True))

        lf1 = DictionaryLabelingFunction('LF_punct', set('!"#$%&*+,./:;<=>?@[\\]^_`{|}~'), 2)
        lf2 = RegexLabelingFunction('LF_punct_2', [re.compile(r'''([&][A-Za-z]+[;]|[*_-]{2,})''', re.I)], 2)
        lfs.append(UnipolarUnionLabelingFunction('LF_punct', [lf1, lf2]))

        lfs.append(RegexEachLabelingFunction('LF_numbers_rgx', [r'''^[-]*[1-9]+[0-9]*([.][0-9]+)*$'''], 2))
        lfs.append(DictionaryLabelingFunction('LF_guidelines', guidelines_dict, 1))


        # =======================================================================
        #
        # UMLS Ontology LFs
        #
        # =======================================================================

        slot_rgxs = [
            #'{} (nebulizer|ointment|inhaler|solution|cream|paste|tablet|syrup|(nitro)*spray|patch|elixer|neb|(([1-9][%] )*(topical )*)*(powder|cream|patch))[s]*',
            '{} (extended|sustained|immediate|immed[.]*|slow) (release|releas|rel[.]*)\s*[)]*',
            '{} enteric coated',
            '{} [(]\s*{}\s*[)]',
            '{} [(]\s*{} (and|[/-]) {}\s*[)]',
            '{} [(]\s*([A-Za-z0-9-\\\]+\s*){{1,3}}\s*[)]',
            '[A-Za-z0-9-]+ [(]\s*{}\s*[)]'
        ]

        for name in sorted(ontologies):
            lf = SlotFilledOntologyLabelingFunction(f'LF_{name}',
                                                    ontologies[name],
                                                    max_ngrams=MAX_NGRAMS,
                                                    slot_patterns=slot_rgxs,
                                                    stopwords=guidelines_sw,
                                                    span_rule=None)
            lfs.append(lf)

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

        #lfs.append(DictionaryLabelingFunction('LF_ctd', ctd_disease_dict, 1, max_ngrams=MAX_NGRAMS, stopwords=guidelines_sw))
        lfs.append(DictionaryLabelingFunction('LF_doid', doid_disease_dict, 2, max_ngrams=MAX_NGRAMS, stopwords=guidelines_sw))
        lfs.append(DictionaryLabelingFunction('LF_hp', hp_disease_dict, 2, max_ngrams=MAX_NGRAMS, stopwords=guidelines_sw))

        # =======================================================================
        #
        # Custom LFs (Regexes, Dictionaries, Heuristics)
        #
        # =======================================================================

        # Custom LFs for this task are more complicated due to i2b2's drug
        # definitions, which includes somewhat ad hoc inclusion criteria
        # of modes, as well as *not* annotating negated, allergy mentions

        # -----------------------------------------------------------------
        # Illicit drugs
        # -----------------------------------------------------------------
        illicit_drugs_dict = open(
            f'{self.data_root}/dictionaries/illicit_drugs.txt',
            'r').read().splitlines()
        lfs.append(
            DictionaryLabelingFunction('LF_illicit_drugs', illicit_drugs_dict,
                                       2, case_sensitive=True))

        # -----------------------------------------------------------------
        # Over-the-counter drugs
        # -----------------------------------------------------------------
        rgx_otc = re.compile(
            r'''\b(mouth wash|eye drop|toe cream|stool softener)[es]*\b''',
            re.I)
        lfs.append(RegexLabelingFunction('LF_otc_rgx', [rgx_otc], 1))

        # -----------------------------------------------------------------
        # Insulin
        # -----------------------------------------------------------------
        rgx_insulin_pos = [
            r'''(lantus [(] insulin glargine [)]|NPH\s*(with)*insulin|insulin (NPH|regular)*\s*(aspart|human)|insulin(\s+(NPH|lispro|regular|regimen|drip|lente|[0-9]+/[0-9]+))*)''',
            r'''(((NPH )*humulin|regular|human|lantus|NPH) )*insulin''',
            r'''(insulin [0-9]+/[0-9]+( human)*)''',
            r'''(insulin\s*NPH|NPH(\s*humulin)*\s*insulin|NPH)''',
        ]
        rgx_insulin_pos = [re.compile(rgx, re.I) for rgx in rgx_insulin_pos]
        lfs.append(
            RegexLabelingFunction('LF_insulin_rgx_1', rgx_insulin_pos, 1))

        # -----------------------------------------------------------------
        # Not insulin mentions
        # -----------------------------------------------------------------
        rgx_insulin_neg = [
            r'''(insulin[- ]dependent)''',
            r'''((non[- ]*)*insulin[- ]dependent([ ,](diabetes mellitus|diabetes|hypothyroidism))*|non[- ]*insulin|insulin sliding scale)'''
        ]
        rgx_insulin_neg = [re.compile(rgx, re.I) for rgx in rgx_insulin_neg]
        lfs.append(
            RegexLabelingFunction('LF_insulin_rgx_2', rgx_insulin_neg, 2))

        # -----------------------------------------------------------------
        # Misc medications
        # -----------------------------------------------------------------
        rgx_drugs = [
            r'''(CCB|KCL)''',
            r'''(vancomycin|levofloxacin|flagyl|iron supplementation)''',
            r'''(KCL\s*(IMMEDIATE|SLOW)\s*(RELEASE|REL[.]*)|asa)''',
            r'''(KCL|potassium chloride)\s*(immediate|slow)\s*(release|rel[.]*)''',
            r'''(chemotherapy|chemo)( regimen[s]*)*''',
            r'''\b(ACEi|ACE INHIBITORS)\b''',
            r'''(metamucil\s*(sugar free)*)''',
            r'''(prednisolone [0-9]{1,2}[%])''',
            r'''(red blood|red blood cell|rbc|platelet) transfusion[s]*''',
            r'''(b|beta)[-]*blocker[s]*''',
            r'''(cardiac|cardiovascular|diabetes|immunosuppressive|pain|copd|(blood\s*)*pressure|oral|outpatient|home|these|your|pressor|pressure)\s*(med(ication|icine)*[s]*)''',
            r'''(metoprolol xl|kcl immediate release)'''
        ]
        rgx_drugs = [re.compile(rgx, re.I) for rgx in rgx_drugs]
        lfs.append(RegexLabelingFunction('LF_drugs_rgx', rgx_drugs, 1))

        # -----------------------------------------------------------------
        # Vitamins
        # -----------------------------------------------------------------
        vit_rgx_1 = re.compile(
            r'''(vitamin|vit[.]*)[s]*[- ][abcde][-]*[1-3]*''', re.I)
        lfs.append(RegexLabelingFunction('LF_vitamins_rgx_1', [vit_rgx_1], 1))

        # -----------------------------------------------------------------
        # Negated vitamins
        # -----------------------------------------------------------------
        vit_rgx_2 = re.compile(
            r'''no ([A-Za-z]+\s*){1,2}(vitamin|vit[.]*)[s]*[- ][abcdek][-]*[1-3]*''',
            re.I)
        lfs.append(RegexLabelingFunction('LF_vitamins_rgx_2', [vit_rgx_2], 2))

        non_drug_concepts_rgxs = [
            # Section Headers
            header_rgx,
            # Lab values e.g., 'phosphate 0.8'
            r'''((?:{})(?: (?:was|of|at))* [0-9]{{1,3}}(?:[.][0-9]+)*[%]*)'''.format(
                '|'.join(sorted(labs, key=len, reverse=1))),
            r'''[0-9]*\s*(to|[-])\s*[0-9]*\s*(red blood cell[s]*)''',
            # Generic mentions of "medications"
            r'''(medication[s]*)''',
            # Drug/chemical mention but not as a prescribed drug
            r'''(potassium chloride policy|(cardiac|adenosine|guanosine)[- ]*mibi|mibi)''',
            # Oxygen saturation
            r'''(o2|oxygen) (saturation|sat)(s)*(\s*[0-9]+[%]*)*''',
            # Allergies
            r'''((allergy|allergies)\s*[:]\s*([A-Za-z]+[ ,]*)+)''',
            r'''(([A-Za-z]+)\s*){1,2}\s*(cause(s|d)*)\s*([A-Za-z]+)*\s*(rash|headache|nausea|swelling|bleeding|reaction)+''',
            # diabetes-related insulin mentions
            r'''((non[- ]*)*insulin[- ]dependent([ ,](diabetes mellitus|diabetes|hypothyroidism))*|non[- ]*insulin|insulin sliding scale)''',
            r'''(diabetes mellitus insulin therapy|insulin toxicity|type 2 diabetes mellitus\s*[,]\s*insulin)''',
            # Proper names
            r'''(coumadin (clinic|program)|oral medicine|captopril renogram test)''',
            # Parenthetical comments
            r'''\( (dose unknown) \)'''
        ]
        non_drug_concepts_rgxs = [re.compile(r, re.I) for r in
                                  non_drug_concepts_rgxs]
        lfs.append(RegexLabelingFunction('LF_drug_concepts_rgx_2',
                                         non_drug_concepts_rgxs, 2))

        # -----------------------------------------------------------------
        # Drug concepts
        # -----------------------------------------------------------------
        dose = r'''[(] [0][.][1-9] (mg) [)]'''
        nitro = r'''(nitroglycerin\s*(((sublingual )*tablet|patch|spray|paste)[s]*)*|nitro[ ]*(spray|patch|paste)|ntg|nitro)[es]*\s*(1/150|(sublingual )*tablet[s]*|spray[s])*'''
        nitro_rgx = f'''({nitro})\s*({dose})*'''

        drug_concepts_rgxs = [
            r'''(metoprolol\s*(tartrate|sustained release)*|toprol[ -]*(xl)*\s*(\( metoprolol\s*(tartrate)* \))*)''',
            r'''toprol xl \( (metoprolol\s*(succinate extended release|\( sust[.] rel[.] \))) \)''',
            r'''(metoprolol xl|kcl immediate release)''',
            r'''(iron|potassium|folate)\s*(supplement(s|ation)*)''',
            r'''(steroids)''',
            r'''(multi|b|maalox|short)[- ](acting|blocker|tablet|vitamin)[s]*''',
            r'''(antibiotic|anti[- ]ischemic|diabetic|torsemide|novolog|medication) regimen''',
            r'''((enteric[- ])*coated\s*(aspirin|asa)|(ecasa|ec asa|enteric coated asa)\s*\( aspirin enteric coated \))''',
            r'''(insulin (7/30|regular|nph) human)''',
            r'''(((cardiac|pain|copd|(blood\s*)*pressure|oral|narcotic pain|outpatient|these|your|pressor|pressure)\s*)(medication[s]*))''',
            r'''((bp|diabetes|sedating|immunosuppressive) (medication|med)[s]*)''',
            nitro_rgx
        ]
        drug_concepts_rgxs = [re.compile(r, re.I) for r in drug_concepts_rgxs]
        lfs.append(
            RegexLabelingFunction('LF_drug_concepts_rgx_1', drug_concepts_rgxs,
                                  1))

        # -----------------------------------------------------------------
        # Head/tail
        # -----------------------------------------------------------------
        # tail of med mentions
        tail_rgx = r'''(human|nebulizer|liquid|sr|mg|release|rel[.]*|patch|liquid|[.]{3})\s*([)]\s*)+'''
        tail_rgx = re.compile(tail_rgx, re.I)

        # head of med mentions
        head_rgx = r'''\s*(XL|MG|REL[.]*|solution|therapeutic|nebulizer|1/150|70/30)\s*([(]\s*)+'''
        head_rgx = re.compile(head_rgx, re.I)

        # 10x longer eval time due to prefix word
        word_rgx = r'''([A-Za-z0-9/.+=-]+\s*){1,5}'''
        nested_rgx = f'''([A-Za-z0-9/.+=-]+\s*)([0-9/]+\s*)*[(]\s*{word_rgx}\s*[(]\s*{word_rgx}\s*[)]\s*[)]'''
        nested_rgx = re.compile(nested_rgx, re.I)

        # slash
        meds_rgx = 'nitrate|lipitor|niacin|calcium|diuril|lasix|plavix|statin|flagyl|ACEi|coum|levo|dig|asa|mag|BB|k'
        slash_rgx = f'''(({meds_rgx})+[/]){{1,3}}({meds_rgx})+'''
        slash_rgx = re.compile(slash_rgx, re.I)

        # punctuation
        punct_rgx = r'''((enteric|ACE)\s*[-]|[.]{3}|((quick )*dissolve|chewable|asa)\s*[/]\s*(chewable)*)'''
        punct_rgx = re.compile(punct_rgx, re.I)

        lfs.append(RegexLabelingFunction('LF_glue_rgx',
                                         [nested_rgx, head_rgx, slash_rgx,
                                          punct_rgx, tail_rgx], 1))

        # TIMEX3 mentions
        lfs.append(RegexLabelingFunction('LF_timex_2', timex.regexes, 2))

        print(f'Labeling Functions n={len(lfs)}')
        return lfs

        #
        # # Illicit drugs
        # illicit_drugs_dict = open(f'{self.data_root}/dictionaries/illicit_drugs.txt','r').read().splitlines()
        # lfs.append(DictionaryLabelingFunction('LF_illicit_drugs', illicit_drugs_dict, 2, case_sensitive=True))
        #
        # # Over-the-counter drugs
        # rgx_otc = re.compile(r'''\b(mouth wash|eye drop|toe cream|stool softener)[es]*\b''')
        # lfs.append(RegexLabelingFunction('LF_otc_rgx', [rgx_otc], 1))
        #
        # # Insulin
        # rgx_insulin_pos = r'''(lantus [(] insulin glargine [)]|NPH\s*(with)*insulin|insulin (NPH|regular)*\s*(aspart|human)|insulin(\s+(NPH|lispro|regular|regimen|drip|lente|[0-9]+/[0-9]+))*|(((NPH )*humulin|regular|human|lantus|NPH) )*insulin)'''
        # lfs.append(RegexLabelingFunction('LF_insulin_rgx_1', [rgx_insulin_pos], 1))
        #
        # # Not insulin mentions
        # rgx_insulin_neg = r'''((non[-]*)*insulin[- ]dependent( diabetes mellitus)*|non[- ]*insulin|insulin sliding scale)'''
        # lfs.append(RegexLabelingFunction('LF_insulin_rgx_2', [rgx_insulin_neg], 2))
        #
        # # Misc medications
        # rgx_drugs = [
        #     r'''(KCL (IMMEDIATE|SLOW) (RELEASE|REL[.]*)|asa)''',
        #     r'''(chemotherapy|chemo)( regimen[s]*)*''',
        #     r'''(red blood|red blood cell|rbc|platelet) transfusion[s]*''',
        #     r'''(b|beta)[-]*blocker[s]*''',
        #     r'''(cardiac|pain|copd|(blood\s*)*pressure|oral|outpatient|home|these|your|pressor|pressure) (med(ication|icine)*[s]*)'''
        # ]
        # rgx_drugs = [re.compile(rgx, re.I) for rgx in rgx_drugs]
        # lfs.append(RegexLabelingFunction('LF_drugs_rgx', rgx_drugs, 1))
        #
        # # Vitamins
        # vit_rgx_1 = re.compile(r'''(vitamin|vit[.]*)[s]*[- ][abcde][-]*[1-3]*''', re.I)
        # lfs.append(RegexLabelingFunction('LF_vitamins_rgx_1', [vit_rgx_1], 1))
        #
        # # Negated vitamins
        # vit_rgx_2 = re.compile(r'''no ([A-Za-z]+\s*){1,2}(vitamin|vit[.]*)[s]*[- ][abcdek][-]*[1-3]*''', re.I)
        # lfs.append(RegexLabelingFunction('LF_vitamins_rgx_2', [vit_rgx_2], 2))
        #
        # # Section Headers
        # lfs.append(RegexLabelingFunction('LF_header_rgx', [header_rgx], 2))
        #
        # # Lab values e.g., 'phosphate 0.8'
        # labs_rgx = r'''((?:{})(?: (?:was|of|at))* [0-9]{{1,3}}(?:[.][0-9]+)*[%]*)'''.format('|'.join(sorted(labs, key=len, reverse=1)))
        # lfs.append(RegexLabelingFunction('LF_labs_rgx', [labs_rgx], 2))
        #
        # # Generic mentions of "medications"
        # meds_rgx = r'''((?!(cardiac|pain|copd|(blood\s*)*pressure|oral|narcotic pain|outpatient|home|these|your|pressor|pressure)\s*)(medication[s]*))'''
        # lfs.append(RegexLabelingFunction('LF_meds_rgx', [re.compile(meds_rgx, re.I)], 2))
        #
        # # Drug/chemical mention but not as a prescribed drug
        # non_drug_rgx = r'''(potassium chloride policy|(cardiac|adenosine|guanosine)[- ]*mibi|mibi)'''
        # lfs.append(RegexLabelingFunction('LF_non_meds_rgx', [re.compile(non_drug_rgx, re.I)], 2))
        #
        # print(f'Labeling Functions n={len(lfs)}')
        # return lfs

###############################################################################
#
# Drug
#
###############################################################################

class DrugLabelingFunctions(object):
    """
    i2b2 labeling functions w/o allergy and negation logic
    """
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
            'receptor',
            'indicator,_reagent,_or_diagnostic_aid',
            'biomedical_or_dental_material',
            'clinical_attribute',
            'enzyme',
            'immunologic_factor',
            'organic_chemical',
            'amino_acid,_peptide,_or_protein',
            'gene_or_genome',
            'amino_acid_sequence',
            'nucleotide_sequence',
            'nucleic_acid,_nucleoside,_or_nucleotide'
        ]

        for sty in ignore:
            if sty in class_map:
                del class_map[sty]

        rm = [sty for sty in class_map if class_map[sty] == 0]
        for sty in rm:
            del class_map[sty]

        return class_map

    def lfs(self, train_sentences, top_k=5, active_tiers=None):

        active_tiers = (1, 2, 3, 4) if active_tiers is None else active_tiers

        # ---------------------------------------------------------------------
        # Supervision Sources
        # ---------------------------------------------------------------------
        print('Loading Ontologies/Dictionaries')
        print('=' * 80)
        sw = set(
            open(f'{self.data_root}/stopwords.txt', 'r').read().splitlines()
        )
        sw = sw.union(set([t[0].upper() + t[1:] for t in sw]))

        # UMLS
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
        if top_k:
            # rank source vocabs by term coverage
            scores = score_umls_ontologies(train_sentences, umls.dictionary)
            rankings = sorted(scores.items(), key=lambda x: x[-1], reverse=1)
            print(f'Matched {len(rankings)} source vocabularies')
            print(f'top_k={top_k} ({top_k / len(rankings) * 100:2.1f}%)')
            sabs, _ = zip(*rankings[0:top_k])
        else:
            # merge all vocabs into a single dictionary
            print('Merge all UMLS source vocabularies')
            sabs = {}

        ontologies = umls_ontology_dicts(sabs, self.class_map, umls)

        # ---------------------------------------------------------------------
        # Labeling Functions
        # ---------------------------------------------------------------------

        reject_headers = {
            'SOCIAL HISTORY', 'ALLERGY', 'ALLERGIES', 'CC', 'DIET',
            'NEUROLOGICAL', 'LABORATORY DATA', 'CHIEF COMPLAINT',
            'FAMILY HISTORY', 'EKG', 'NEURO', 'PAST SURGICAL HISTORY',
            'FAMILY HISTORY', 'ALLG', 'LABORATORY', 'PREOPERATIVE LABS',
            'LABS'}
        header_rgx = r'''^({})[:]'''.format(
            '|'.join(sorted(reject_headers, key=len, reverse=1))
        )

        negex = NegEx(data_root=f'{self.data_root}/negex/')

        guidelines_sw = {'tobacco', 'alcohol', 'water', 'blood pressure'}

        # custom stopwords
        # These prevent biases from the main ontologies using the independent snorkel model.
        # For example, "blood pressure medications", blood pressure is a negative concept in all UMLS ontologies
        # and we don't capture the chain dep linking it to medications by default.
        guidelines_sw = guidelines_sw.union(
            {'NAD', 'nad', 'fat', 'sugar', 'PCP', 'pcp',
             'chemo', 'chemotherapy', 'lipase', 'amylase',
             'smoke', 'mg', 'NPH', 'nph', 'INH',
             'calcium', 'magnesium', 'iron', 'potassium',
             'glucose', 'phosphate', 'electrolyte', 'salt',
             'meds', 'medications', 'medication', 'drug', 'gentle',
             'tomorrow', 'dose', 'unknown', 'air', 'toxin', 'prevent',
             'increase', 'subcutaneous', 'intravenous', 'diurese',
             'pain', 'blood pressure', 'oral', 'coated', 'enteric',
             'topical', 'immediate', 'release', 'slow', 'mg',
             'mercury', 'basis', 'FLAIR', 'products', 'hypoglycemic',
             'blood', 'drip', 'at', 'each', 'daily',
             'red blood cells', 'immediate', 'release',
             'lead', 'leads', 'nitrites', 'nitrite', 'stent', 'milligrams',
             'MIBI', 'mibi', 'Mibi', 'regimen', 'adenosine', 'Adenosine MIBI',
             'subcutaneous', 'unfractionated', 'intravenous', 'O2',
             })

        # Common lab values (i.e., not drugs)
        labs = {
            'hemoglobin', 'sodium', 'bicarbonate', 'phosphorus', 'magnesium',
            'lactate', 'PCO2', 'white count', 'lipase',
            'platelets', 'alk phos', 'white blood cell count', 'phosphate',
            'MB', 'PT', 'serum osms', 'AST', 'amylase',
            'CO2', 'potassium', 'carbon dioxide', 'creatinine', 'glucose',
            'troponin', 'PTT', 'white blood cell[s]*',
            'BUN', 'calcium', 'WBC', 'chloride', 'bicarb', 'ALT',
            'O2 saturation', 'beta hydroxybutyrate', 'anion gap',
            'TSH', 'lipase', 'PO2', 'platelet count', 'CK', 'hematocrit',
            'BNP', 'ABG', 'T[.]* bili', 'INR', 'ABG T',
            '[TD]-bili', 'T-bili', 'D-bili', 'o2 sat', 'o2 staturation', 'hCG',
            'O2 sat', 'O2 SAT', 'o2 saturation',
            'T3', 'T4', 'acetone', 'coagulation factors', 'lactic acid', 'PTH'
        }

        guidelines_sw = guidelines_sw.union(set(labs))
        guidelines_sw = guidelines_sw.union({t + 's' for t in guidelines_sw})

        MAX_NGRAMS = 8

        # Preliminary.Annotation.Guidelines.6.12.pdf
        fpath = f'{self.data_root}/dictionaries/guidelines/i2b2-guidelines-drugs.txt'
        guidelines_dict = open(fpath,'r').read().splitlines()
        guidelines_dict = set(guidelines_dict).union({t.lower() for t in guidelines_dict})

        lfs = []

        # =======================================================================
        #
        # Baseline LFs
        #
        # =======================================================================

        lfs.append(DictionaryLabelingFunction('LF_stopwords', sw, 2, case_sensitive=True))
        lfs.append(DictionaryLabelingFunction('LF_punct', set('!"#$%&*+,./:;<=>?@[\\]^_`{|}~'), 2))
        lfs.append(RegexEachLabelingFunction('LF_numbers_rgx', [r'''^[-]*[1-9]+[0-9]*([.][0-9]+)*$'''], 2))
        lfs.append(DictionaryLabelingFunction('LF_guidelines', guidelines_dict, 1))

        # =======================================================================
        #
        # UMLS Ontology LFs
        #
        # =======================================================================

        slot_rgxs = [
            #'{} (nebulizer|ointment|inhaler|solution|cream|paste|tablet|syrup|(nitro)*spray|patch|elixer|neb|(([1-9][%] )*(topical )*)*(powder|cream|patch))[s]*',
            '{} (extended|sust[.]|sustained|immediate|immed[.]*|slow) (release|releas|rel[.]*)\s*[)]*',
            '{} (enteric coated|enteric\s*[-]\s*coated)',
            '{} [(]\s*{}\s*[)]',
            '{} [(]\s*{} (and|[/-]) {}\s*[)]',
            '{} [(]\s*([A-Za-z0-9-\\\]+\s*){{1,3}}\s*[)]',
            '[A-Za-z0-9-]+ [(]\s*{}\s*[)]'
        ]

        for name in sorted(ontologies):
            lf = SlotFilledOntologyLabelingFunction(f'LF_{name}',
                                                    ontologies[name],
                                                    max_ngrams=MAX_NGRAMS,
                                                    slot_patterns=slot_rgxs,
                                                    stopwords=guidelines_sw,
                                                    span_rule=None)
            lfs.append(lf)

        # Schwartz-Hearst abbreviations
        lfs.append(AbbrvDefsLabelingFunction('LF_schwartz_hearst_abbrvs_1', class_dictionaries[1], 1))
        lfs.append(AbbrvDefsLabelingFunction('LF_schwartz_hearst_abbrvs_2', class_dictionaries[2], 2))

        # =======================================================================
        #
        # External Ontologies
        #
        # =======================================================================

        lfs.append(TermMapLabelingFunction('LF_specialist_synset_1', specialist_1, 1))
        lfs.append(TermMapLabelingFunction('LF_specialist_synset_2', specialist_2, 2))
        lfs.append(TermMapLabelingFunction('LF_adam_synset_1', adam_synset_1, 1))
        lfs.append(TermMapLabelingFunction('LF_adam_synset_2', adam_synset_2, 2))

        lfs.append(DictionaryLabelingFunction('LF_ctd', ctd_chemical_dict, 1, max_ngrams=MAX_NGRAMS, stopwords=guidelines_sw))
        lfs.append(DictionaryLabelingFunction('LF_doid', doid_disease_dict, 2, max_ngrams=MAX_NGRAMS, stopwords=guidelines_sw))
        lfs.append(DictionaryLabelingFunction('LF_hp', hp_disease_dict, 2, max_ngrams=MAX_NGRAMS, stopwords=guidelines_sw))

        if 4 not in active_tiers:
            print(f'Labeling Functions n={len(lfs)}')
            return lfs

        # =======================================================================
        #
        # Custom LFs (Regexes, Dictionaries, Heuristics)
        #
        # =======================================================================

        # -----------------------------------------------------------------
        # Illicit drugs
        # -----------------------------------------------------------------
        illicit_drugs_dict = open(f'{self.data_root}/dictionaries/illicit_drugs.txt', 'r').read().splitlines()
        lfs.append(DictionaryLabelingFunction('LF_illicit_drugs', illicit_drugs_dict, 2, case_sensitive=True))

        # -----------------------------------------------------------------
        # Over-the-counter drugs
        # -----------------------------------------------------------------
        rgx_otc = re.compile(
            r'''\b(mouth wash|eye drop|toe cream|stool softener)[es]*\b''')
        lfs.append(RegexLabelingFunction('LF_otc_rgx', [rgx_otc], 1))

        # -----------------------------------------------------------------
        # Insulin
        # -----------------------------------------------------------------
        rgx_insulin_pos = [
            r'''(lantus [(] insulin glargine [)]|NPH\s*(with)*insulin|insulin (NPH|regular)*\s*(aspart|human)|insulin(\s+(NPH|lispro|regular|regimen|drip|lente|[0-9]+/[0-9]+))*)''',
            r'''(((NPH )*humulin|regular|human|lantus|NPH) )*insulin''',
            r'''(insulin [0-9]+/[0-9]+( human)*)''',
            r'''(insulin\s*NPH|NPH(\s*humulin)*\s*insulin|NPH)''',
        ]
        lfs.append(
            RegexLabelingFunction('LF_insulin_rgx_1', rgx_insulin_pos, 1))

        # -----------------------------------------------------------------
        # Not insulin mentions
        # -----------------------------------------------------------------
        rgx_insulin_neg = r'''((non[- ]*)*insulin[- ]dependent([ ,](diabetes mellitus|diabetes|hypothyroidism))*|non[- ]*insulin|insulin sliding scale)'''
        lfs.append(
            RegexLabelingFunction('LF_insulin_rgx_2', [rgx_insulin_neg], 2))

        # -----------------------------------------------------------------
        # Misc medications
        # -----------------------------------------------------------------
        rgx_drugs = [
            r'''(CCB|KCL|BB)''',
            r'''(vancomycin|levofloxacin|flagyl|iron supplementation)''',
            r'''(KCL\s*(IMMEDIATE|SLOW)\s*(RELEASE|REL[.]*)|asa)''',
            r'''(KCL|potassium chloride)\s*(immediate|slow)\s*(release|rel[.]*)''',
            r'''(chemotherapy|chemo)( regimen[s]*)*''',
            r'''\b(ACEi|ACE INHIBITORS)\b''',
            r'''(metamucil\s*(sugar free)*)''',
            r'''(prednisolone [0-9]{1,2}[%])''',
            r'''(red blood|red blood cell|rbc|platelet) transfusion[s]*''',
            r'''(b|beta)[-]*blocker[s]*''',
            r'''(cardiac|cardiovascular|diabetes|immunosuppressive|pain|copd|(blood\s*)*pressure|oral|outpatient|home|these|your|pressor|pressure)\s*(med(ication|icine)*[s]*)''',
            r'''(metoprolol xl|kcl immediate release)'''
        ]
        rgx_drugs = [re.compile(rgx, re.I) for rgx in rgx_drugs]
        lfs.append(RegexLabelingFunction('LF_drugs_rgx', rgx_drugs, 1))

        # -----------------------------------------------------------------
        # Vitamins
        # -----------------------------------------------------------------
        vit_rgx_1 = re.compile(
            r'''(vitamin|vit[.]*)[s]*[- ][abcde][-]*[1-3]*''', re.I)
        lfs.append(RegexLabelingFunction('LF_vitamins_rgx_1', [vit_rgx_1], 1))

        # -----------------------------------------------------------------
        # Negated vitamins
        # -----------------------------------------------------------------
        vit_rgx_2 = re.compile(
            r'''no ([A-Za-z]+\s*){1,2}(vitamin|vit[.]*)[s]*[- ][abcdek][-]*[1-3]*''',
            re.I)
        lfs.append(RegexLabelingFunction('LF_vitamins_rgx_2', [vit_rgx_2], 2))

        # -----------------------------------------------------------------
        # Non drug mentions
        # -----------------------------------------------------------------
        non_drug_concepts_rgxs = [
            # Section Headers
            header_rgx,
            # Lab values e.g., 'phosphate 0.8'
            r'''((?:{})(?: (?:was|of|at))* [0-9]{{1,3}}(?:[.][0-9]+)*[%]*)'''.format(
                '|'.join(sorted(labs, key=len, reverse=1))),
            r'''[0-9]*\s*(to|[-])\s*[0-9]*\s*(red blood cell[s]*)''',
            # Generic mentions of "medications"
            r'''(medication[s]*)''',
            # Drug/chemical mention but not as a prescribed drug
            r'''(potassium chloride policy|(cardiac|adenosine|guanosine)[- ]*mibi|mibi)''',
            # Oxygen saturation
            r'''(o2|oxygen) (saturation|sat)(s)*(\s*[0-9]+[%]*)*''',
            # Allergies
            r'''((allergy|allergies)\s*[:]\s*([A-Za-z]+[ ,]*)+)''',
            r'''(([A-Za-z]+)\s*){1,2}\s*(cause(s|d)*)\s*([A-Za-z]+)*\s*(rash|headache|nausea|swelling|bleeding|reaction)+''',
            # diabetes-related insulin mentions
            r'''((non[- ]*)*insulin[- ]dependent([ ,](diabetes mellitus|diabetes|hypothyroidism))*|non[- ]*insulin|insulin sliding scale)''',
            r'''(diabetes mellitus insulin therapy|insulin toxicity|type 2 diabetes mellitus\s*[,]\s*insulin)''',
            # Proper names
            r'''(coumadin (clinic|program)|oral medicine|captopril renogram test)''',
            # Parenthetical comments
            r'''\( (dose unknown) \)'''
        ]
        non_drug_concepts_rgxs = [re.compile(r, re.I) for r in
                                  non_drug_concepts_rgxs]
        lfs.append(RegexLabelingFunction('LF_drug_concepts_rgx_2',
                                         non_drug_concepts_rgxs, 2))

        # -----------------------------------------------------------------
        # Drug concepts
        # -----------------------------------------------------------------
        dose = r'''[(] [0][.][1-9] (mg) [)]'''
        nitro = r'''(nitroglycerin\s*(((sublingual )*tablet|patch|spray|paste)[s]*)*|nitro[ ]*(spray|patch|paste)|ntg|nitro)[es]*\s*(1/150|(sublingual )*tablet[s]*|spray[s])*'''
        nitro_rgx = f'''({nitro})\s*({dose})*'''

        drug_concepts_rgxs = [
            r'''(metoprolol\s*(tartrate|sustained release)*|toprol[ -]*(xl)*\s*(\( metoprolol\s*(tartrate)* \))*)''',
            r'''toprol xl \( (metoprolol\s*(succinate extended release|\( sust[.] rel[.] \))) \)''',
            r'''(metoprolol xl|kcl immediate release)''',
            r'''(iron|potassium|folate)\s*(supplement(s|ation)*)''',
            r'''(steroids|chemotherapy)''',
            r'''(multi|b|maalox|short)[- ](acting|blocker|tablet|vitamin)[s]*''',
            r'''(antibiotic|anti[- ]ischemic|diabetic|torsemide|novolog|medication) regimen''',
            r'''((enteric[- ])*coated\s*(aspirin|asa)|(ecasa|ec asa|enteric coated asa)\s*\( aspirin enteric coated \))''',
            r'''(insulin (7/30|regular|nph) human)''',
            r'''(((cardiac|pain|copd|(blood\s*)*pressure|oral|narcotic pain|outpatient|these|your|pressor|pressure)\s*)(medication[s]*))''',
            r'''((bp|diabetes|sedating|immunosuppressive) (medication|med)[s]*)''',
            nitro_rgx
        ]
        drug_concepts_rgxs = [re.compile(r, re.I) for r in drug_concepts_rgxs]
        lfs.append(RegexLabelingFunction('LF_drug_concepts_rgx_1', drug_concepts_rgxs, 1))

        # -----------------------------------------------------------------
        # Head/Root/Tail punctuation
        # -----------------------------------------------------------------
        # tail of med mentions
        tail_rgx = r'''(human|nebulizer|liquid|sr|mg|release|rel[.]*|patch|liquid|[.]{3})\s*([)]\s*)+'''
        tail_rgx = re.compile(tail_rgx, re.I)

        # head of med mentions
        head_rgx = r'''\s*(XL|MG|REL[.]*|solution|therapeutic|nebulizer|1/150|70/30)\s*([(]\s*)+'''
        head_rgx = re.compile(head_rgx, re.I)

        punct_rgx = r'''((enteric|ACE)\s*[-]|[.]{3}|(dissolve|chewable|asa)\s*[/])'''
        punct_rgx = re.compile(punct_rgx, re.I)
        lfs.append(RegexLabelingFunction('LF_glue_rgx',
                                         [head_rgx, punct_rgx, tail_rgx], 1))

        print(f'Labeling Functions n={len(lfs)}')
        return lfs
