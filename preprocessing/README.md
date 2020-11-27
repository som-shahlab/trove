# Clinical Text NLP Pre-processing
This pipeline uses spaCy for all NLP parsing. Pipes for clinical text
tokenization and sentence boundary detection (SBD) are designed to be fast and
accurate enough for within-sentence relational inference.

## 1. Instructions
For fast loading with Pandas, documents should be in TSV format. This format
minimally requires 2 named columns: `DOC_NAME` and `TEXT`; all additional columns are treated as document metadata and added to the `metadata` field in the final output JSON file.

If your input is a directory of text files, you can convert to TSV files
of size `batch_size` using:

	python notes_to_tsv.py \
		--inputdir <INDIR> \
		--outputdir <OUTDIR> \
		--batch_size 5000

Parse TSV files and dump to a JSON container format:

	python parse.py \
		--inputdir <INDIR> \
		--outputdir <OUTDIR> \
		--prefix mimic \
		--n_procs 4 \
		--disable ner,tagger,parser \
		--batch_size 10000
		
You should set batch size based on the number of CPU cores you plan on using during processing. Currently load balancing is done manually, so partition your data into some multiple of our `n_procs` while keeping `batch_size` large enought to keep your CPUs busy.

## 2. Benchmarks

### Sub-sample
- 50,000 MIMIC-III documents
- 4 core MacBook Pro 2.5Ghz mid-2015

| Time (minutes) | Disable Pipes | NLP Output |
|---------------|----------------|------------|
| 1.5 | `ner,parser,tagger` | tokens, SBD|
| 5.6 | `ner,parser` | tokens, SBD, POS tags|
| 17.9 | `ner` | tokens, SBD, POS tags, dependency tree |


### All of MIMIC-III v1.4
- 2,083,180 MIMIC-III Documents
- 16 cores

| Time (minutes) | Disable Pipes | NLP Output |
|---------------|----------------|------------|
| 28.6 | `ner,parser,tagger` | tokens, SBD|
| 52.4 | `ner,parser` | tokens, SBD, POS tags|



## 3. JSON Output Format
The JSON format consists for a document name, a sentence offset index `i`, a list of sentences with NLP markup (e.g., POS tags), and (optional) document metadata.

```
{
"name":"7569_NURSING_OTHER_REPORT_1992373",
"sentences":[{
	"words":["CCU","Progress","Note",":","S","-","intubated","&","sedated","."],
	"abs_char_offsets":[2,6,15,19,22,23,25,35,37,44],
	"i":0}, ...],
"metadata": {...}
}
```