#!/bin/bash

n_model_search=250
tier=123
task="chemical"

#indir="/Users/fries/Desktop/label_matrices_inkfish_final/chemical/BC5CDR/${tier}/"
#outdir="/Users/fries/Desktop/label_matrices_inkfish_final/RESULTS/${task}/"
#outdir="/Users/fries/Desktop/label-model-scale/${task}/"

indir="/Users/fries/Desktop/label_matrices_inkfish_BIO/chemical/BC5CDR/${tier}/"
outdir="/Users/fries/Desktop/label_model_BIO/${task}/"

#seeds=(1234 8888 2020 1701 9090)
#top_k=(0 1 2 3 4 5 6 7 8 9 19 49 100)

seeds=(1234 8888 2020 1701 9090)
#seeds=(126 24 316 450 541)
# best tier 1234
#top_k=(4)

# best tier 123
top_k=(0)

for k in "${top_k[@]}"
do
  for seed in "${seeds[@]}"
  do
    echo "Running k=${k} seed=${seed} | ${indir} > ${task}.k${k}.tier${tier}.seed${seed}.log"
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
