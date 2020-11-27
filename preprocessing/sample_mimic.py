import os
import sys
import random
import datetime
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from prep_mimic import preprocess


def load_clinical_notes(row_ids, mimic_fpath):

	n_matches = len(row_ids)
	docs = {}
	for chunk in pd.read_csv(mimic_fpath,
							 sep=',',
							 compression='infer',
							 chunksize=10000):
		for row in chunk.itertuples():
			try:
				if row.ROW_ID in row_ids:
					docs[row.ROW_ID] = row
					n_matches -= 1
			except Exception as e:
				print(e, row.ROW_ID)

		if n_matches <= 0:
			print(f'Matched all {len(row_ids)} row ids')
			break
	return docs


def load_row_ids(fpath):
	return set([int(x) for x in open(fpath, 'r').read().splitlines()])


def process_clinical_notes(docs, preserve_offsets=True):
	for row_id in docs:
		row = docs[row_id]
		row = {name:getattr(row, name) for name in row._fields if name != 'Index'}
		
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
		row['DOC_NAME'] = f"{row['ROW_ID']}_{row['SUBJECT_ID']}_{row['HADM_ID']}"

		# convert note text
		text, tdelta = preprocess(row["TEXT"],
								  doc_ts=doc_ts,
								  preserve_offsets=preserve_offsets)
		# escape whitespace
		text = text.replace('\n', '\\n').replace('\t', '\\t').replace('\r', '\\r') 
		
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
		row['CHARTTIME'] = str(charttime)  if charttime is not None else np.nan  
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


def reservoir_sampling(iterable, n, seed=1234):
	np.random.seed(seed)
	i = 0
	pool = []
	for item in iterable:
		if len(pool) < n:
			pool.append(item)
		else:
			i += 1
			k = random.randint(0, i)
			if k < n:
				pool[k] = item
	return pool


def main(args):
	# MIMIC-III v1.4 has 2,083,180 note records
	samples = set(np.random.choice(range(2083180),
								   args.sample_size,
								   replace=False))

	print("Loading MIMIC-III notes ...")
	documents = load_clinical_notes(samples, args.infile)
	print(f"...loaded {len(documents)} documents.")

	print("Processing MIMIC-III notes...")
	process_clinical_notes(documents, preserve_offsets=True)

	print("Writing TSV output ...")
	dump_tsvs(documents, args.outputdir)


if __name__ == '__main__':

	argparser = argparse.ArgumentParser()
	argparser.add_argument("-i", "--infile", type=str, required=True,
						   help="path to NOTEEVENTS.csv.gz file")
	argparser.add_argument("-n", "--sample_size", type=int, default=10000)
	argparser.add_argument("-o", "--outputdir", type=str, required=True,
						   help="output directory")
	
	args = argparser.parse_args()

	main(args)
