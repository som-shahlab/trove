#!/bin/bash
# apply labeling functions

n_procs=4
indir="/Users/fries/Desktop/NER-Datasets-ALL/"
#outdir="/Users/fries/Desktop/label_matrices_inkfish_final/chemical/BC5CDR/"
outdir="/Users/fries/Desktop/label_matrices_inkfish_BIO/chemical/BC5CDR/"

dataroot="/Users/fries/Code/refactor/inkNER/data/supervision/"
unlabeled="/Users/fries/Desktop/PUBMED-UNLABELED/pubmed-pfizer/json/"

task="chemical"
corpus="cdr"
domain="pubmed"
tag_fmt="BIO"

#top_k=(0 1 2 3 4 5 6 7 8 9 19 49 100)
#tiers=(1234 123 12 1)

#tiers=(1234)
#top_k=(4)

tiers=(123)
top_k=(0)

#tiers=(12 1)
#top_k=(9)

for tier in "${tiers[@]}"
do
for k in "${top_k[@]}"
do
  echo "${task} | Top k: ${k} | tiers: ${tier}"
  python ../../apply-lfs.py \
  --indir ${indir} \
  --outdir ${outdir}/${tier}/k${k}/ \
  --data_root ${dataroot} \
  --corpus ${corpus} \
  --domain ${domain} \
  --tag_fmt ${tag_fmt} \
  --entity_type ${task} \
  --top_k ${k} \
  --active_tiers ${tier} \
  --n_procs ${n_procs} > ${task}.${corpus}.${domain}.${tag_fmt}.${k}.${tier}.log
done
done

  #--unlabeled ${unlabeled} \
  #--n_sample_docs 1000 \
