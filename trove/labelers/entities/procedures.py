import re
from trove.labelers.tools import *
from trove.labelers.labeling import *
from trove.utils import score_umls_ontologies
from trove.labelers.norm import lowercase, strip_affixes
from trove.labelers.abbreviations import AbbrvDefsLabelingFunction
from trove.labelers.umls import umls_ontology_dicts, load_sem_groups


###############################################################################
#
# Procedures
#
###############################################################################

class ProcedureLabelingFunctions(object):
    """

    TODO This needs work!


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

        procedures = [
            'therapeutic_or_preventive_procedure',
            'diagnostic_procedure',
            'laboratory_procedure',
            'molecular_biology_research_technique'
        ]
        class_map = {sab: 1 for sab in procedures}

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

        # ---------------------------------------------------------------------
        # Load Resources
        # ---------------------------------------------------------------------
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

        # Create class dictionaries (i.e., the union of all sab/stys)
        class_dictionaries = collections.defaultdict(dict)
        for sab, sty in umls.dictionary:
            if sty in self.class_map:
                class_dictionaries[self.class_map[sty]].update(
                    dict.fromkeys(umls.dictionary[(sab, sty)]))

        print("Class dictionaries")
        for label in class_dictionaries:
            print(f'y={label} n={len(class_dictionaries[label])}')

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

        lfs = []
        MAX_NGRAMS = 5

        for name in sorted(ontologies):
            lfs.append(OntologyLabelingFunction(f'LF_{name}',
                                          ontologies[name],
                                          max_ngrams=MAX_NGRAMS,
                                          stopwords=sw))

        lfs.append(AbbrvDefsLabelingFunction('LF_schwartz_hearst_abbrvs_1', class_dictionaries[1], 1, stopwords=sw))
        lfs.append(AbbrvDefsLabelingFunction('LF_schwartz_hearst_abbrvs_2', class_dictionaries[2], 2, stopwords=sw))

        lfs.append(TermMapLabelingFunction('LF_specialist_synset_1', specialist_1, 1, stopwords=sw))
        lfs.append(TermMapLabelingFunction('LF_specialist_synset_2', specialist_2, 2, stopwords=sw))
        lfs.append(TermMapLabelingFunction('LF_adam_synset_1', adam_synset_1, 1, stopwords=sw))
        lfs.append(TermMapLabelingFunction('LF_adam_synset_2', adam_synset_2, 2, stopwords=sw))