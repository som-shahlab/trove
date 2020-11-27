import sys
#sys.path.insert(0,'../../ehr-rwe/')

import os
import glob
import time
import argparse

from trove.data.dataloaders import dataloader
from rwe.utils import load_dict
from rwe.utils import build_candidate_set

from trove.labelers import TaggerPipelineServer

from trove.labelers.taggers import (
    ResetTags, DocTimeTagger, PrecomputedEntityTagger,
    DictionaryTagger, HypotheticalTagger, HistoricalTagger,
    SectionHeaderTagger, ParentSectionTagger,
    Timex3Tagger, Timex3NormalizerTagger, TimeDeltaTagger,
    FamilyTagger, PolarityTagger, TextFieldDocTimeTagger
)

def timeit(f):
    """
    Decorator for timing function calls
    :param f:
    :return:
    """
    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        print(f'{f.__name__} took: {te - ts:2.4f} sec')
        return result

    return timed

def dump_concepts(documents, target_concepts, outfpath='concepts.tsv'):
    """Dump CSV of concepts"""
    header = ['DOC_ID', 'DOC_TS', 'TYPE', 'TEXT', 'ABS_CHAR_START', 'ABS_CHAR_END']
    header += ['POLARITY', 'HYPOTHETICAL', 'HISTORICAL', 'SECTION', 'SUBJECT', 'TDELTA']
    data = []
    for entity_type in target_concepts:
        for doc in documents:
            spans = build_candidate_set([doc], entity_type)
            for x in spans:
                row = [doc.name, doc.props['doctime'] if 'doctime' in doc.props else 'None', entity_type]
                row += [x.text, x.abs_char_start, x.abs_char_end]

                polarity = x.props['polarity'] if 'polarity' in x.props else 'NULL'
                hypothetical = x.props['hypothetical'] == 1 if 'hypothetical' in x.props else 'NULL'
                historical = x.props['historical'] == 1 if 'historical' in x.props else 'NULL'
                section = x.props['section'].text if 'section' in x.props and x.props['section'] is not None else 'NULL'
                subject = x.props['subject'] if 'subject' in x.props else 'NULL'
                tdelta = x.props['tdelta'] if 'tdelta' in x.props else 'NULL'

                row += [polarity, hypothetical, historical, section, subject, tdelta]
                data.append('\t'.join(map(str, row)))

    with open(outfpath, 'w') as fp:
        fp.write('\t'.join(header) + '\n')
        fp.write('\n'.join(data))

def get_header_dict():
    return [
        'Allergen Reactions',
        'Attending Attestations',
        'Attending Attestions',
        'Chief Complaint',
        'Clinical Decision Rules',
        'Diagnosis Code',
        'ED Course, Data Review & Interpretation',
        'ED Treatment',
        'Family History',
        'HPI',
        'History & Physical',
        'History From Shared Lists',
        'Labs & Imaging',
        'Labs ordered',
        'Medical Decision Making',
        'Medications',
        'New Prescriptions',
        'Occupational History',
        'Past Medical History',
        'Patient Active Problem List',
        'Physical Exam',
        'Prior to Admission Medications',
        'Procedures',
        'Recent Labs',
        'Review of Systems',
        'Social History',
        'Substance and Sexual Activity',
        'Summary of assessment',
        'Tobacco Use',
        'Critical Care and Sepsis',
        'Critical Care and Sepsis Alert',
        'Ultrasounds & Procedures',
        'Reason for Hospitalization',
        'Patient Active Problem List'
    ]


def get_major_headers():
    return [
        'Clinical Decision Rules',
        'Diagnosis Code',
        'ED Course, Data Review & Interpretation',
        'ED Treatment',
        'Family History',
        'HPI',
        'History & Physical',
        'Labs & Imaging',
        'Medical Decision Making',
        'New Prescriptions',
        'Past Medical History',
        'Physical Exam',
        'Prior to Admission Medications',
        'Procedures',
        'Review of Systems',
        'Social History',
        'Summary of assessment'
    ]

@timeit
def main(args):

    # =========================================================================
    # Load Parsed Documents
    # =========================================================================
    if os.path.isdir(args.input):
        filelist = glob.glob(f'{args.input}/*.json')
    else:
        filelist = [args.input]
    print(f'Loading {len(filelist)} files')
    corpus = [dataloader(filelist)]
    print(f'Documents: {len(corpus[0])}')

    # =========================================================================
    # Define Concept Pipeline
    # =========================================================================

    # SNOMED dictionaries with
    if args.concepts == "umls":
        dict_disorder = load_dict(
            f'{args.dict_root}viruses/SNOMEDCT_US.disorder.tsv')
        dict_symptom = load_dict(
            f'{args.dict_root}viruses/SNOMEDCT_US.symptom.tsv')
        dict_finding = load_dict(
            f'{args.dict_root}viruses/SNOMEDCT_US.finding.tsv')
        dict_icd10 = load_dict(f'{args.dict_root}viruses/ICD10CM.codes.tsv')
        dict_geo = load_dict(
            f'{args.dict_root}viruses/umls.geographic_area.tsv')

        taggers = {
            "concepts": DictionaryTagger({
                'disorder': dict_disorder,
                'symptom': dict_symptom,
                'finding': dict_finding,
                'GPE': dict_geo,
                'ICD10': dict_icd10})
        }
        target_entities = ['disorder', 'symptom', 'finding', 'ICD10']

    # Merge all dictonaries into a single entity type
    elif args.concepts == 'umls_merged':
        dict_terms = {}
        dict_terms.update(
            load_dict(f'{args.dict_root}viruses/SNOMEDCT_US.disorder.tsv'))
        dict_terms.update(
            load_dict(f'{args.dict_root}viruses/SNOMEDCT_US.symptom.tsv'))
        dict_terms.update(
            load_dict(f'{args.dict_root}viruses/SNOMEDCT_US.finding.tsv'))

        taggers = {"concepts": DictionaryTagger(
            {'disorder_symptom_finding': dict_terms})}
        target_entities = ['disorder_symptom_finding']
        print(f'[{args.concepts}] Loaded {len(dict_terms)} concept terms')

    # Precomputed entities (we use weakly supervised entities here)
    elif args.concepts == 'trove':

        drug_fpath = f'{args.entity_tags}/drug.tags.tsv'
        diso_fpath = f'{args.entity_tags}/disorder.tags.tsv'

        dict_icd10 = load_dict(f'{args.dict_root}viruses/ICD10CM.codes.tsv')
        dict_geo = load_dict(
            f'{args.dict_root}viruses/umls.geographic_area.tsv')
        taggers = {
            "concepts": DictionaryTagger(
                {'GPE': dict_geo, 'ICD10': dict_icd10}),
            "drugs": PrecomputedEntityTagger(drug_fpath,
                                             type_name='drug'),
            "disorders": PrecomputedEntityTagger(diso_fpath,
                                                 type_name='disorder')
        }
        target_entities = ['disorder', 'drug', 'ICD10']

    # Entity/Concept Pipeline
    pipeline = {
        "headers": SectionHeaderTagger(header_dict=get_header_dict(),
                                       stop_headers={})
    }

    # Concepts
    for name in taggers:
        pipeline[name] = taggers[name]
    pipeline["timex3"] = Timex3Tagger()

    # Attributes
    attribs = {
        # canonicalize datetimes
        "doctimes": DocTimeTagger(prop='CREATED_AT',
                                  format='%Y-%m-%d %H:%M:%S'),
        "normalize": Timex3NormalizerTagger(),

        # concept modifiers
        "section": ParentSectionTagger(targets=target_entities + ['TIMEX3'],
                                       major_headers=get_major_headers()),
        "tdelta": TimeDeltaTagger(targets=target_entities),
        "polarity": PolarityTagger(targets=target_entities,
                                   data_root=f"{args.dict_root}/negex/"),
        "hypothetical": HypotheticalTagger(targets=target_entities),
        "historical": HistoricalTagger(targets=target_entities),
        "subject": FamilyTagger(targets=target_entities,
                                data_root=f"{args.dict_root}/negex/")
    }
    pipeline.update(attribs)
    print(pipeline.keys())
    print(f'Pipes: {len(pipeline)}')

    # =========================================================================
    # Run Tagging Pipeline & Dump Concepts
    # =========================================================================
    tagger = TaggerPipelineServer(num_workers=args.n_procs)
    documents = tagger.apply(pipeline, corpus)
    print('Tagging complete')

    dump_concepts(documents[0],
                  target_concepts=['disorder', 'drug', 'ICD10', 'GPE'],
                  outfpath=args.output)

    print(f'Concepts written to {args.output}')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None, required=True)
    parser.add_argument("--output", type=str, default=None, required=True)
    parser.add_argument("--dict_root", type=str, default='data/supervision/dicts/')
    parser.add_argument("--entity_tags", type=str, default=None)
    parser.add_argument("--n_procs", type=int, default=16)
    parser.add_argument("--concepts", type=str, default="umls_merged")
    args = parser.parse_args()

    main(args)


