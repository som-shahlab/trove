# Trove
[![DOI](https://zenodo.org/badge/316359795.svg)](https://zenodo.org/badge/latestdoi/316359795)
### NOTE: This is branch is depricated and being refactored into master

 
Tools for weakly supervised sequence labeling and span classification in biomedical text. Code is provided for training and evaluating Snorkel models for unsupervised ensembling of dictionary and other heuristic methods of labeling text.
 
**Benchmark Tasks**

 - 4 NER tasks from scientific literature (BC5CDR) and EHRs (ShARe/CLEF, i2b2) for *Disease, Chemical, Disorder, Drug* entities
 - 2 span classification (ShARe/CLEF, THYME) (*Negation*, *Temporality*) 

Each NER task is evaluated under different ablation settings assuming increasing degrees of available labeling resources. Tiers are additive.

1. Annotation Guideline Rules
2. UMLS Ontologies 
3. 3rd Party Ontologies / Text-mined Lexicons
4. Task-specific Labeling Functions

## I. Quick Start

Download datasets and dictionaries dependencies [here](https://drive.google.com/drive/folders/1zeZPZUBlV-jh-WCDK8YnkIU3Pm-LSqwu?usp=sharing)

This assumes your documents have been preprocessed into a JSON container format. 

### 1. Generate Label Matrices 
```
python apply-lfs.py \
--indir <INDIR> \
--outdir <OUTDIR> \
--data_root <DATA> \
--corpus cdr \
--domain pubmed \
--tag_fmt IO \
--entity_type chemical \
--top_k 4 \
--active_tiers 1234 \
--n_procs 4 > chemical.lfs.log
```

### 2. Train Label Model
Use the default Snorkel label model to 

```
python train-label-model.py \
--indir <INDIR> \
--outdir <OUTDIR> \
--train 0 \
--dev 1 \
--test 2 \
--n_model_search 50 \
--tag_fmt_ckpnt BIO \
--seed 1234 > chemical.label_model.log
```

### 3. Train End Model (e.g., BERT) 

```
python proba-train-bert.py \
--train chemical.train.prior_balance.proba.conll.tsv \
--dev chemical.dev.prior_balance.proba.conll.tsv \
--test chemical.test.prior_balance.proba.conll.tsv \
--model biobert_v1.1_pubmed-pytorch/ \
--device cuda \
--n_epochs 10 \
--lr 1e-5 \
--seed 1234 > chemical.bert.log
```

