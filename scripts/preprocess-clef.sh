#
# Build MIMIC-III & ShARe/CLEF clinical notes datasets
#
#
#

# Sample 50,000 documents from MIMIC-III

#python ../preprocessing/parse.py \
#	--inputdir  ../data/corpora/unlabeled/ \
#	--outputdir  ../data/corpora/ \
#	--prefix mimic_unlabeled \
#	--n_procs 4 \
#	--disable ner,tagger,parser \
#	--batch_size 5000

#
# ShARe/CLEF
#

CLEF_HOME="/Users/fries/Desktop/CLEF-FINAL/release/docs/"
OUTPUT="/Users/fries/Desktop/output/"

# transform docs into TSV (preprocess mimic is required)
python ../preprocessing/notes_to_tsv.py \
--inputdir "${CLEF_HOME}" \
--outputdir "${OUTPUT}/clef/" \
--batch_size 5000 \
--preprocess mimic

# NLP preprocessing # ,tagger,parser
python ../preprocessing/parse.py \
--inputdir "${OUTPUT}/clef/" \
--outputdir "${OUTPUT}" \
--prefix clef \
--n_procs 4 \
--disable ner \
--batch_size 10000

# transform dates and PHI blinding
python preprocessing/preprocess_shareclef.py \
--infile "${OUTPUT}/clef.0.json" \
--outfile "${OUTPUT}/shareclef.json"

# delete intermediary files
rm "${OUTPUT}/clef.0.json"
rm -fr "${OUTPUT}/clef/"


