"""
ShARe/CLEF docs have a header

387055	||||	26563	||||	18146	||||	RADIOLOGY_REPORT	||||	2009-01-26 15:08:00.0	||||	C12 CHEST (PORTABLE AP)	||||		||||	Clip # 282-0776  Actual report	||||

?|?|?|CATEGORY|CHARTDATE|?|?|Clip #

MIMIC-III v1.4
CATEGORY	CGID	CHARTDATE	CHARTTIME	DESCRIPTION	DOC_NAME	HADM_ID	ISERROR	ROW_ID	STORETIME	SUBJECT_ID	TEXT

"""
import os
import sys
import glob
import random
import datetime
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from prep_mimic import preprocess


def process_clinical_notes(docs):
    for row_id in docs:
        row = docs[row_id]
        row = {name: getattr(row, name) for name in row._fields if
               name != 'Index'}

        # convert timestamps
        chartdate = None if type(row['CHARTDATE']) is not str \
            else datetime.strptime(row['CHARTDATE'], '%Y-%m-%d')
        charttime = None if type(row['CHARTTIME']) is not str \
            else datetime.strptime(row['CHARTTIME'], '%Y-%m-%d %H:%M:%S')

        # get structured chart time
        doc_ts = charttime.year if charttime else chartdate.year

        if type(row['HADM_ID']) is not str and np.isnan(row['HADM_ID']):
            row['HADM_ID'] = 'NaN'
        else:
            row['HADM_ID'] = int(row['HADM_ID'])

        row['SUBJECT_ID'] = int(row['SUBJECT_ID'])
        row['ROW_ID'] = int(row['ROW_ID'])
        row[
            'DOC_NAME'] = f"{row['ROW_ID']}_{row['SUBJECT_ID']}_{row['HADM_ID']}"

        # convert note text
        text, tdelta = preprocess(row["TEXT"], doc_ts=doc_ts,
                                  preserve_offsets=True)
        # escape whitespace
        text = text.replace('\n', '\\n').replace('\t', '\\t').replace('\r',
                                                                      '\\r')

        # if timedelta is 0, then no full datetimes were found in the note,
        if tdelta == 0:
            sample_range = range(2008, 2020)
            tdelta = int(doc_ts - np.random.choice(sample_range, 1)[0])

        if chartdate:
            chartdate -= timedelta(days=tdelta * 365)
        if charttime:
            charttime -= timedelta(days=tdelta * 365)

        row['TEXT'] = text
        row['CHARTDATE'] = str(chartdate.date())
        row['CHARTTIME'] = str(charttime) if charttime is not None else np.nan
        docs[row_id] = row




def dump_tsvs(documents, fpath):
    with open(f'{fpath}/mimic-n{len(documents)}.tsv', 'w') as fp:
        for i, row_id in enumerate(documents):
            row = documents[row_id]
            header = sorted(row.keys())
            if i == 0:
                fp.write('\t'.join(header))
                fp.write('\n')
            values = [str(row[col]) for col in header]
            line = '\t'.join(values)
            fp.write(f'{line}\n')


def main(args):

    documents = glob.glob(f'{args.inputdir}/*.tsv')
    pass

    # print("Loading MIMIC-III notes ...")
    # documents = load_clinical_notes(samples, args.infile)
    # print(f"...loaded {len(documents)} documents.")
    #
    # print("Processing MIMIC-III notes...")
    # process_clinical_notes(documents)
    #
    # print("Writing TSV output ...")
    # dump_tsvs(documents, args.outputdir)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-i", "--infile", type=str, required=True,
                           help="path to NOTEEVENTS.csv.gz file")
    argparser.add_argument("-n", "--sample_size", type=int, default=10000)
    argparser.add_argument("-o", "--outputdir", type=str, required=True,
                           help="output directory")

    args = argparser.parse_args()

    main(args)
