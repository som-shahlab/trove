#!/bin/bash
# apply labeling functions

n_procs=4
indir="/Users/fries/Desktop/NER-Datasets-ALL/"
#outdir="/Users/fries/Desktop/label_matrices_inkfish_final/i2b2meds/"
outdir="/Users/fries/Desktop/label_matrices_inkfish_BIO/i2b2meds/"

dataroot="/Users/fries/Code/refactor/inkNER/data/supervision/"
unlabeled="/Users/fries/Desktop/NER-Datasets-ALL/ehr/i2b2meds/unlabeled/"

task="drug"
corpus="i2b2meds"
domain="ehr"
tag_fmt="BIO"

#top_k=(0 1 2 3 4 5 6 7 8 9 19 49 100)

# best for tier 1234
top_k=(2)
tiers=(1234)

# best for tier 123
top_k=(1)
tiers=(123)

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
