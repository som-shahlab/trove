#!/bin/bash

tier=$1
n_model_search=100
task="disease"

indir="/Users/fries/Desktop/label_matrices_inkfish_final/disease/BC5CDR/${tier}/"
outdir="/Users/fries/Desktop/label_matrices_inkfish_final/RESULTS/${task}/"

seeds=(1234 8888 2020 1701 9090)
top_k=(0) # 1 2 3 4 5 6 7 8 9 19 49 100)

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
      --seed ${seed} > ${task}.k${k}.tier${tier}.seed${seed}.log
    done
done

