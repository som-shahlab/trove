import re
import functools
import itertools
import collections
import pandas as pd

import collections
from trove.labelers.tools import *
from trove.labelers.labeling import *
from trove.utils import score_umls_ontologies
from trove.labelers.norm import lowercase, strip_affixes
from trove.labelers.abbreviations import AbbrvDefsLabelingFunction
from trove.labelers.umls import (
    umls_ontology_dicts,
    load_sem_groups,
    umls_classmap_dicts
)
from pytorch_pretrained_bert import BertTokenizer


def bert_tokenizer(text, tokenizer):
    tokens = tokenizer.tokenize(text)
    toks = []

    curr = ""
    for t in tokens:
        if t[0:2] == '##':
            curr = curr + t[2:]
        elif curr:
            toks.append(curr)
            curr = t
        elif not curr:
            curr = t
        else:
            toks.append(t)
            curr = ""
    if curr:
        toks.append(curr)

    return toks

def build_word_graph(dictionary, tokenizer, min_occur=50):

    G = collections.defaultdict(collections.Counter)
    for text in dictionary:
        tokens = bert_tokenizer(text, tokenizer)
        if len(tokens) == 1:
            continue
        for i in range(len(tokens)-1):
            G[tokens[i]][tokens[i+1]] += 1

    if min_occur:
        for head in G:
            rm = []
            for tail in G[head]:
                if G[head][tail] < min_occur:
                    rm.append(tail)
            for tail in rm:
                del G[head][tail]
    return dict(G)

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


    def lfs(self, train_sentences, top_k=10, active_tiers=None):
        """

        1) Guidelines
        2) UMLS
        3) 3rd Party Dictionaries
        4) Manual Regexes, Rules, Dictionaries

        From this we define the following resource levels
        High (1,2,3,4)
        Med (1,2)
        Low (1,4)

        Parameters
        ----------
        train_sentences
        top_k
        umls_as_ontologies

        Returns
        -------

        """
        active_tiers = (1, 2, 3, 4) if active_tiers is None else active_tiers

        print('Loading Ontologies/Dictionaries')
        print('=' * 80)
        sw = set(
            open(f'{self.data_root}/stopwords.txt', 'r').read().splitlines()
        )
        sw = sw.union(set([t[0].upper() + t[1:] for t in sw]))

        # Unified Medical Language System
        umls = load_umls(f'{self.data_root}/ontologies/umls2018AA.parquet', sw)

        print('(Source Vocabulary/Semantic Type) Pairs',len(umls.dictionary))
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
                                              umls,3
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
            top_k = min(len(rankings), top_k)
            print(f'top_k={top_k} ({top_k/len(rankings)*100:2.1f}%)')
            sabs, _ = zip(*rankings[0:top_k])
            print("rankings:", sabs)
            sabs = list(sabs) + ['OTHER']
        else:
            # merge all vocabs into a single dictionary
            print('Merge all UMLS source vocabularies')
            sabs = []

        #lf_dicts = umls_classmap_dicts(sabs, self.class_map, umls)
        ontologies = umls_ontology_dicts(sabs, self.class_map, umls)



        # ---------------------------------------------------------------------
        # Labeling Functions
        # ---------------------------------------------------------------------
        lfs = []

        # Task guidelines provide example annotations
        # https://biocreative.bioinformatics.udel.edu/media/store/files/2015/bc5_CDR_data_guidelines.pdf
        # Page 6 "Do not annotate"
        guidelines = 'cdr-chemicals-guidelines-neg-expanded.txt' \
            if 4 in active_tiers else 'cdr-chemicals-guidelines-neg.txt'

        fpath = f'{self.data_root}/dictionaries/guidelines/{guidelines}'

        guidelines_sw = open(fpath,'r').read().splitlines()
        guidelines_sw = set(guidelines_sw).union(
            {t + 's' for t in guidelines_sw}
        )

        # more custom stopwords
        if 4 in active_tiers:
            guidelines_sw = guidelines_sw.union({
                'Hg', 'lead', 'TG', 'DS', 'TMA', 'CN', 'AT', 'tobacco',
                'species', 'leads', 'LPO', 'at', 'fluorescent', 'BMCs', 'BMC',
                'angiotensinogen', 'buffer', 'reactive', 'received', 'direct',
                'releasing', 'deliver', 'free'
            })

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

        # Page 6
        # SELECT STR FROM MRCONSO WHERE CODE in (D004967, D000928, D003276)
        special_cases = {
            'estrogen', 'estrogen receptor agonist', 'estrogenic agent',
            'estrogenic compound', 'estrogenic effect', 'antidepressant',
            'antidepressive agent', 'antidepressant drug', 'thymoanaleptics',
            'thymoleptics', 'low-dose oral contraceptive',
            'oral contraceptive', 'phasic oral contraceptive'
        }
        special_cases = special_cases.union({t + 's' for t in special_cases})

        # Page 5 of bc5_CDR_data_guidelines
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

        # don't include any guideline term mentions
        if 1 not in active_tiers:
            guidelines_dict = {}

        # build word graphs
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased',
                                                  do_lower_case=False)
        G_umls  = build_word_graph(class_dictionaries[1], tokenizer, min_occur=10)
        G_chebi = build_word_graph(chebi_dict, tokenizer, min_occur=10)
        G_ctd   = build_word_graph(ctd_chemical_dict, tokenizer, min_occur=10)

        # labels for stopwords are defined by dictionaries
        guidelines_sw = {t: 2 for t in guidelines_sw}

        # =======================================================================
        #
        # Baseline LFs
        #
        # =======================================================================
        MAX_NGRAMS = 8

        if 1 in active_tiers:
            # guideline labeling functions
            lfs.append(DictionaryLabelingFunction('LF_guidelines_1', guidelines_dict, 1, max_ngrams=MAX_NGRAMS))
            lfs.append(DictionaryLabelingFunction('LF_guidelines_2', guidelines_sw, 2, max_ngrams=MAX_NGRAMS))
            lfs.append(DictionaryLabelingFunction('LF_special_cases', special_cases, 1, max_ngrams=MAX_NGRAMS))

            # stopwords, numbers, and punctuation.
            lfs.append(DictionaryLabelingFunction('LF_stopwords', sw, 2, max_ngrams=MAX_NGRAMS, case_sensitive=True))
            lfs.append(DictionaryLabelingFunction('LF_punct', set('!"#$%&*+,./:;<=>?@[\\]^_`{|}~'), 2))
            lfs.append(RegexEachLabelingFunction('LF_numbers_rgx', [r'''^[-]*[1-9]+[0-9]*([.][0-9]+)*$'''], 2))


        # =======================================================================
        #
        # UMLS Ontology LFs
        #
        # =======================================================================

        if 2 in active_tiers:
            # UMLS ontologies
            #for name in sorted(ontologies):
            for name in sabs:
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
        if 3 in active_tiers:
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
        if 4 in active_tiers:

            not_chems_rgx = [
                r'''\b([A-Za-z0-9]+?[rlntd]ase[s]*)\b''',
                r'''[A-Za-z0-9]+ factor[s]*''',
                r'''(angiotensinogen)''',
                # some suppliments aren't annotated
                r'''(guarana|(panax )*ginseng)''',
                r'''\b(anti[a-z]+)\b''',
                r'''(cocaine (abuse|addiction|overdose))''',
                r'''((renal)\s*(angiotensinogen) (mRNA|expression))''',
                r'''(atrial natriuretic factor( \[\s*ANF\s*\])*)''',
                r'''(fibrinolysis inhibitor[s]*)''',
                r'''((brain )*biogenic amines)''',
                # r'''((unfractionated|subcutaneous) heparin( injection[s]*))''',
                r'''([-]\s*(associated|dependent|related|treated|acting|controlled|induced|containing|fold|increasing|adjusted|month|specific))'''
            ]
            proteins_rgx = [re.compile(rgx, re.I) for rgx in not_chems_rgx]
            lfs.append(RegexLabelingFunction('LF_not_chemicals_rgx', not_chems_rgx, 2))

            # ----------------------------------------------------------------------
            # misc chemicals
            # ----------------------------------------------------------------------
            rgxs = [
                r'''(oral contraceptives)''',
                r'''([A-Z]){2}[0-9]{3,}''',
                r'''\b(ACEi|ACE inhibitor[s]*)\b''',
                r'''(corticosteroid[s]*|(oral[- ])*contraceptive[s]*)''',
                r'''(calcium|cacl[(]2[)])''',
                r'''([l][- ](glutathione|arginine))''',
                r'''(appetite[- ]suppressant[s]*( drugs)*)''',
                r'''(calcium channel blocker[s]*|calcium chloride|CaCl)''',
                r'''(simvastatin[- ]ezetimibe)''',
                r'''([snp][-](perillyl alcohol|pyrimidinyl|choloroaniline|acetylcysteine|limonene))''',
                r'''((alpha|beta|gamma)[-][T])''',
                re.compile(r'''(PG[-]9|U[-]II)'''),
                re.compile(r'''(BPO|GSH|DFU|CsA|Srl|HOE|GVG|PAN|NMDA)'''),
                re.compile(r'''(TCR|MZ|HBsAg|AraG|LR132|SSRI[s]*|HBeAg|LR132|BD10[0-9]{2}|GNC92H2|SSR103800|CGRP)'''),
                # peptides and proteins with less than 15 amino acids ARE annotated
                r'''(angiotensin([- ]ii)*)''',
                r'''(u[- ]ii|urotensin[- ]ii)''',
                # a peptide (11 amino acids in humans)
                r'''bradykinin''',  # 9 amino acids
                r'''(d[- ]pen(icillamine)*)'''  # ?,
                r'''(lipopolysaccharide|alkylating agents)''',
                r'''(pegylated (interferon|IFN)( alpha[- ]2[ab])*)''',
                r'''(\[3H\])''',
                r'''(CaCl|LAM|GSH|PAN|H2O|AVP|LR132)'''
            ]
            rgxs = [re.compile(rgx, re.I) if type(rgx) is str else rgx for rgx in rgxs]
            lfs.append(RegexLabelingFunction('LF_misc_rgx', rgxs, 1))

            # ----------------------------------------------------------------------
            # abbreviation errors from the non-UMLS resource and drug class mentions
            # ----------------------------------------------------------------------
            abbrvs_dict = {
                'mRNA', 'Pgp', 'TMA', 'ATT', 'BMC', 'BMCs', 'CR', 'AED', 'CO', 'BZD',
                'HR', 'RAS', 'CN', 'ALT', 'BUN', 'CPK', 'HEM', 'TAA',
            }
            term_dict = {t for t in drug_class_dict}
            term_dict.update(abbrvs_dict)
            lfs.append(DictionaryLabelingFunction('LF_drug_class_and_abbrvs', term_dict, 2, max_ngrams=1))

            # ----------------------------------------------------------------------
            # glue chemical terms together
            # ----------------------------------------------------------------------
            glue_rgxs = [
                r'''(\[[0-9][-.])''', # [1-
                r'''(\[\s*3H\s*\])''', # [3H]
                r'''(thiazolyl|amino|phenyl|ethyl|butyl|nonane|3H)[\]]''',
                r'''([-](methyl|ethyl|carboline|dimethoxy|alpha|beta|delta|gamma|glyceryl|thiazolyl)[-])''',
                re.compile(r'''[-]([1-9]|[A-Z])[-]'''),
            ]
            glue_rgxs = [re.compile(rgx, re.I) if type(rgx) is str else rgx for rgx in glue_rgxs]
            lfs.append(RegexLabelingFunction('LF_glue_rgx', glue_rgxs, 1))

            # ----------------------------------------------------------------------
            # Hyphen token
            # ----------------------------------------------------------------------
            def get_subtokens(dictionary, split_chars=['-'], min_occur=20):
                freq = collections.Counter()
                for term in dictionary:
                    for ch in split_chars:
                        n = term.count(ch)
                        if n >= 1:
                            for tok in term.split('-'):
                                freq[tok] +=1

                subtoks = {}
                for item in sorted(freq.items(), key=lambda x:x[-1], reverse=1):
                    t,n = item
                    if not t.strip():
                        continue
                    t = t.strip()
                    if n < min_occur:
                        break
                    subtoks[t] = n
                return subtoks

            subtoks = get_subtokens(ctd_disease_dict)
            subtokens = sorted(subtoks.keys(), key=lambda x: len(x), reverse=1)
            subtokens = [re.escape(t) for t in subtokens]
            chemical_rgx = f"(({'|'.join(subtokens)})[-]){{2,}}"
            chemical_rgx = re.compile(chemical_rgx, re.I)
            lfs.append(RegexLabelingFunction('LF_grammar_ctd_rgx', [chemical_rgx], 1))

            # ----------------------------------------------------------------------
            # Parentheses tokens
            # ----------------------------------------------------------------------
            parens_rgxs = [
                r'''[(](P|p|n)\s*([><=]+|(less|great)(er)*)|(ml|mg|kg|g|(year|day|month)[s]*)[)]|[(][0-9]+[%][)]'''
            ]
            parens_rgxs = [re.compile(rgx, re.I) for rgx in parens_rgxs]
            lfs.append(
                RegexLabelingFunction('LF_parentheses_rgx', parens_rgxs, 2))

            # ----------------------------------------------------------------------
            # Word graphs
            # ----------------------------------------------------------------------
            stopwords = {'day', 'dose', 'mg', 'ml', '%', 'long',
                         'non', 'mL', '.', '/','dL', 'high', ';',
                         'kg', 'intra', 'wave', '+', 'to', 'a'}
            stopwords.update({t.lower() for t in sw})

            lfs.append(WordGraphLabelingFunction('LF_umls_word_graph', G_umls,
                                                 label=1, min_length=10,
                                                 sw=stopwords))
            lfs.append(
                WordGraphLabelingFunction('LF_chebi_word_graph', G_chebi,
                                          label=1, min_length=10,
                                          sw=stopwords))
            lfs.append(
                WordGraphLabelingFunction('LF_ctd_word_graph', G_ctd, label=1,
                                          min_length=10, sw=stopwords))


        print(f'Labeling Functions n={len(lfs)}')
        return lfs






