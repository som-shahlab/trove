#!/bin/bash

n_model_search=100
tier=123
task="disease"

indir="/Users/fries/Desktop/label_matrices_inkfish_BIO/${task}/BC5CDR/${tier}/"
outdir="/Users/fries/Desktop/label_model_BIO/${task}/"

#seeds=(1234 8888 2020 1701 9090)
seeds=(126 24 316 450 541)

# best tier 1234, 123
top_k=(0)

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
