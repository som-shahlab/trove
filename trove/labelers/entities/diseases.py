import re
import string
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
# Disease (BC5CDR)
#
###############################################################################
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

        # Forcing abstains on these types just lower majority vote
        # and don't improve label model performance
        # ignore = [
        #     'body_location_or_region',
        #     'body_part,_organ,_or_organ_component',
        #     'body_space_or_junction'
        # ]

        for sty in ignore:
            if sty in class_map:
                del class_map[sty]

        rm = [sty for sty in class_map if class_map[sty] == 0]
        for sty in rm:
            del class_map[sty]

        return class_map

    def lfs(self, train_sentences, top_k=10, active_tiers=None):
        """

        Parameters
        ----------
        train_sentences
        top_k
        active_lf_tiers
            Few-shot/annotation guidelines (1)
            KBs (1,2,3)
            All Resources (1,2,3,4)
            No KBs (1,4)


        Returns
        -------

        """
        active_tiers = (1,2,3,4) if active_tiers is None else active_tiers

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
        # special_cases_dict = {
        #     'akinetic', 'bleed', 'consolidation', 'fall', 'lesion',
        #     'malalignment', 'mental status', 'opacification', 'opacities',
        #     'swelling', 'tamponade', 'unresponsive', 'LVH', 'NAD', 'PNA',
        #     'DNR', 'SSCP'
        # }

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
            sabs = list(sabs) + ['OTHER']
        else:
            # merge all vocabs into a single dictionary
            print('Merge all UMLS source vocabularies')
            sabs = {}

        ontologies = umls_ontology_dicts(sabs, self.class_map, umls)

        # # Fix special cases
        # for name in sorted(ontologies):
        #     for term in special_cases_dict:
        #         if term in ontologies[name]:
        #             ontologies[name][term] = [1., 0.]

        # ---------------------------------------------------------------------
        # Word Graphs
        # ---------------------------------------------------------------------
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased',
                                                  do_lower_case=False)
        ctd_doid_hp_dict = {}
        ctd_doid_hp_dict.update(dict.fromkeys(ctd_disease_dict))
        ctd_doid_hp_dict.update(dict.fromkeys(doid_disease_dict))
        ctd_doid_hp_dict.update(dict.fromkeys(hp_disease_dict))

        G_umls = build_word_graph(class_dictionaries[1], tokenizer,
                                  min_occur=10)
        G_ctd_doid_hp = build_word_graph(ctd_doid_hp_dict, tokenizer,
                                         min_occur=10)

        # ---------------------------------------------------------------------
        # Labeling Functions
        # ---------------------------------------------------------------------
        lfs = []

        # Task guidelines provide example annotations
        # https://biocreative.bioinformatics.udel.edu/media/store/files/2015/bc5_CDR_data_guidelines.pdf

        # use custom guideline stopwords not explicitly mentioned in the docs
        if 4 in active_tiers:
            guidelines = 'cdr-diseases-guidelines-neg-expanded.txt'
        else:
            guidelines = 'cdr-diseases-guidelines-neg.txt'

        fpath = f'{self.data_root}/dictionaries/guidelines/{guidelines}'

        guidelines_sw = open(fpath, 'r').read().splitlines()
        guidelines_sw = set(guidelines_sw).union(
            {t + 's' for t in guidelines_sw}
        )
        fpath = f'{self.data_root}/dictionaries/guidelines/cdr-guidelines-diseases.txt'
        guidelines_dict = open(fpath, 'r').read().splitlines()

        guidelines_sw = {t: 2 for t in guidelines_sw}

        # =====================================================================
        #
        # Baseline LFs
        #
        # =====================================================================
        if 1 in active_tiers:
            lfs.append(DictionaryLabelingFunction('LF_stopwords', sw, 2, case_sensitive=True))
            #lfs.append(DictionaryLabelingFunction('LF_punct',set('!"#$%&*+,./:;<=>?@[\\]^_`{|}~'), 2))
            lfs.append(DictionaryLabelingFunction('LF_punct', set('!()"#$%&*+,./:;<=>?@[\\]^_`{|}~'), 2))

            lfs.append(RegexEachLabelingFunction('LF_numbers_rgx', [r'''^[-]*[1-9]+[0-9]*([.][0-9]+)*$'''], 2))
            lfs.append(DictionaryLabelingFunction('LF_guidelines_1', guidelines_dict, 1))
            lfs.append(DictionaryLabelingFunction('LF_guidelines_2', guidelines_sw, 2))

        # =====================================================================
        #
        # UMLS Ontology LFs
        #
        # =====================================================================
        if 2 in active_tiers:
            # UMLS ontologies
            for name in sorted(ontologies):
            #for name in sabs:
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
        if 3 in active_tiers:
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
        if 4 in active_tiers:

            # ------------------------------------------------------------------------------------------
            # Char grams
            # ------------------------------------------------------------------------------------------
            lfs.append(RegexEachLabelingFunction('LF_bio_ch_grams_rgx',
                                     [r'''^(psych|necro|nephro|hyper|throm|hypo|acro|hemo)[A-Za-z]+?([rlt]ic)$''',
                                      r'''^(hepato|hemato|nephro|cardio|neuro|myelo|oto)*toxic(ities|ity)*$'''], 1))

            # ------------------------------------------------------------------------------------------
            # Regex Disease Concept Patterns
            # ------------------------------------------------------------------------------------------
            anatomy = {'liver', 'central nervous system', 'brain', 'aorta', 'oral cavity', 'bladder', 'artery', 'ureter'}
            finding = {'calcification', 'disorders', 'carcinoma', 'cirrhosis', 'cancer', 'injury', 'enlargement', 'angiosarcoma', 'adenocarcinoma'}
            anat_finding_rgx = '({}) (in|of) the ({})'.format('|'.join(sorted(finding, key=len, reverse=1)),  '|'.join(sorted(anatomy, key=len, reverse=1)))

            anat_dict = {'renal','myocardial','axonal','neuronal','kidney','bladder','cardiac','hippocampal',
                'mitochondrial','cord','hepatic','cerebellum', 'hepatic', 'tissue', 'liver', 'hepatocellular',
                'myocardial( cell)*','proteinuric','axonal', 'tissue'}

            regexes = [
                r'''(([A-Za-z]+\s*) and ([A-Za-z]+\s*) ((neuro)*toxicity|injury|lesion[s]*|impairment|effusion[s]*|deficit[s]*))''',
                r'''([A-Za-z]+) and ([A-Za-z]+) (insufficiency|(dysfunction|carcinoma|cancer|syndrome|disorder|disease)[s]*)''',
                # hyphens
                r'''\b(non[-](small|hodgkin)|veno[-]occlusive|end[-]stage|HBV[-]HIV|Q[-]T)\b''',
                # increases/decreases in X
                r'''(increase[s]* in (blood pressure|heart rate|locomotor activity|dural( and cortical) blood flow))''',
                r'''((reduction|decrease)[s*] in (MAP|glomerular number|(arterial )*blood pressure))''',
                r'''((respiratory|hypothalamic|corticostriatal|tubular|biventricular|myocardial|hepatic|systolic|cranial nerve|sexual) dysfunction[s]*)''',
                # injuries
                r'''({}) (injury|damage)'''.format(
                    '|'.join(sorted(anat_dict, key=len, reverse=1))),
                r'''(malignant ([A-Za-z]+ )*(glioma|tumor)[s]*)''',
                r'''(([A-Za-z]+)'s|wolff[- ]+parkinson[- ]+white|haemolytic[- ]+uraemic|guillain[- ]+barr|hematologic|cholestatic|rabbit)([- ]+like)* syndrome''',
                r'''diabetic( hyperalgesia)*|diabetes''',
                # anatomy findings
                anat_finding_rgx,
                # common disease patterns
                r'''\b[A-Za-z-]+'s (syndrome|disease)\b''',
                r'''((artery )*calcification)|(calcification of the [A-Za-z]+)''',
                # dystrophy  (5/20)
                r"""([Dd]uchenne('s)* (muscular )*dystrophy|DMD)""",
                # common findings
                r'''(ventricular tachyarrhythmias|loss of consciousness|tachyarrhythmias|hyperhidrosis|hypertensive|cardiomegaly|weight gain|hypotension|weight loss|glucosuria|hoarding)''',
            ]
            regexes =[re.compile(rgx, re.I) if type(rgx) is str else rgx for rgx in regexes]
            lfs.append(RegexLabelingFunction('LF_misc_diseases_rgx_1', regexes, 1))

            # 6/6/2020 - Added after inital round of error analysis
            more_regexes = [
                r'''(hyperactive|convulsive|haemorrhage|depressed|deformation[s]*)''',
                r'''\b((sugar|drug) dependency|nicotine-induced nystagmus|nystagmus|NIN)\b''',
                # movement/muscule issues
                r'''(weakness of extremities|transverse limb deficiency|increase in locomotor activity|palpebral twitching)''',
                r'''(choreoathetoid movement[s]*|choreatiform hyperkinesias)''',
                r'''(tender joints|tenderness|swelling|morning stiffness|excessive flexion)''',
                # cardiac
                r'''(valve|valvular|valvular heart) (regurgitation|abnormalit(y|ies))''',
                r'''(atherosclerotic obstruction|cardiac remodelling)''',
                # neurological/renal
                r'''(cholestatic|renovascular|renal and kidney) disease[s]*''',
                r'''(cranial nerve|hepatic and renal|cardiac|renal) dysfunction[s]*''',
                r'''(neuronal loss|cranial nerve deficits|hippocampal injury|behavioral abnormalities|deficits in communication|repetitive behaviors|impaired immediate free recall)''',
                r'''(vanishing bile duct|renal and hepatic failure|hepatic impairment|deterioration of renal function|abnormal liver function)''',
            ]
            more_regexes = [re.compile(rgx, re.I) if type(rgx) is str else rgx
                            for rgx in more_regexes]
            lfs.append(RegexLabelingFunction('LF_more_diseases_rgx_1', more_regexes, 1))

            regexes = [
                # drug induced / associated effects aren't labeled
                r'''([-]\s*(associated|dependent|related|treated|acting|controlled|induced|containing|fold|increasing|adjusted|month|specific))''',
                # toxic effects aren't diseases
                r'''(toxic ((side )*effect[s]*|agent[s]*|action|state|reaction|range|death[s]*|profile|assault[s]*)|(highly|minimally) toxic)'''
            ]
            regexes = [re.compile(rgx, re.I) if type(rgx) is str else rgx for
                       rgx in regexes]
            lfs.append(
                RegexLabelingFunction('LF_misc_diseases_rgx_2', regexes, 2))

            # ------------------------------------------------------------------------------------------
            # Word Graphs
            # ------------------------------------------------------------------------------------------

            stopwords = {t for t in sw}
            stopwords.update({t.lower() for t in sw})
            stopwords.update({'caused', 'without', 'induced', 'pre', 'treatment',
                              'associated', 'following',
                              'day','days', 'inhibitor', 'enzyme', 'imaging',
                              'inhibitors', 'agents', 'agent',
                              'systolic', 'related', 'due', 'factor',
                              'significant', 'blockers', 'blocker',
                              'system', 'membrane', 'recurrent', 'excessive',
                              'cells', 'focal', 'central', 'premature',
                              'severe', 'organic', 'upper', 'de novo', 'de',
                              'novo', 'right', 'left', 'reductase', 'adverse'
                             })
            stopwords.update(list(string.punctuation))
            stopwords.update(list(string.digits))

            lfs.append(WordGraphLabelingFunction('LF_umls_word_graph', G_umls, label = 1, min_length=3, sw = stopwords))
            lfs.append(WordGraphLabelingFunction('LF_ctd_doid_hp_word_graph', G_ctd_doid_hp, label = 1, min_length=3, sw = stopwords))

        print(f'Labeling Functions n={len(lfs)}')
        return lfs