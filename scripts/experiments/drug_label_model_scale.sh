#!/bin/bash

n_model_search=250
#tier=1234
tier=123
task="drug"

#indir="/Users/fries/Desktop/label_matrices_inkfish_final/i2b2meds/${tier}/"
#outdir="/Users/fries/Desktop/label_matrices_inkfish_final/RESULTS/${task}/"


indir="/Users/fries/Desktop/label_matrices_inkfish_BIO/i2b2meds/${tier}/"
outdir="/Users/fries/Desktop/label_model_BIO/i2b2meds/"

#seeds=(1234)

seeds=(126 24 316 450 541)
#top_k=(2)
top_k=(1)

for k in "${top_k[@]}"
do
  for seed in "${seeds[@]}"
  do
    echo "Running k=${k} seed=${seed} | ${indir}"
	  python ../../train-label-model.py \
	  --indir ${indir}k${k} \
      --outdir ${outdir}/${tier}/${k} \
	  --train 0 \
	  --dev 1 \
	  --test 2 \
	  --n_model_search ${n_model_search} \
      --tag_fmt_ckpnt IO \
      --seed ${seed} > ${task}.k${k}.tier${tier}.seed${seed}.log
	done
done
