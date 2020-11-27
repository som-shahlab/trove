#!/bin/bash

n_model_search=100
tier=1234
task="drug"

indir="/Users/fries/Desktop/label_matrices_inkfish_final/i2b2meds/${tier}/"
outdir="/Users/fries/Desktop/label_matrices_inkfish_final/DEV_SUBSAMPLE_RESULTS/${task}/"

seeds=(1234 8888 2020 1701 9090)
data_seeds=(0 1000 6501 2500 6666)

n_docs=(1 2 3 4 5 10 15 25 50)
k=2

for n in "${n_docs[@]}"
do
  for seed in "${seeds[@]}"
  do
    for dseed in "${data_seeds[@]}"
    do
    echo "Running doc_n=${n} tierl=${tier} seed=${seed} dataseed=${dseed} | ${indir}"
	  python ../../train-label-model.py \
	  --indir ${indir}k${k} \
      --outdir ${outdir}/${tier}/${k} \
	  --train 0 \
	  --dev 1 \
	  --test 2 \
      --n_dev_docs ${n} \
	  --n_model_search ${n_model_search} \
	  --seed ${seed} \
      --data_seed ${dseed} > ${task}.valid${n}.${k}.${seed}.${dseed}.log
	done
  done
done
