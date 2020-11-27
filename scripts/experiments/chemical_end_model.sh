#!/bin/bash

task="chemical"
indir="/share/pi/nigam/projects/jfries/inkfish_proba_conll_final/"
model="/share/pi/nigam/projects/jfries/biobert_v1.1_pubmed-pytorch/"

seeds=(1234 8888 2020 1701 9090)
n_epochs=30
lr=2e5
labels="prior_balance"

# weakly supervised runs
for seed in "${seeds[@]}"
do
python proba-train-bert-final.py \
    --train ${indir}/${task}/${task}.train.prior_balance.proba.conll.tsv \
    --dev ${indir}/${task}/${task}.dev.prior_balance.proba.conll.tsv \
    --test ${indir}/${task}/${task}.test.prior_balance.proba.conll.tsv \
    --model ${model} \
    --device cuda \
    --n_epochs ${n_epochs} \
    --lr ${lr} \
    --seed ${seed} > ${task}.${labels}.seed_${seed}.log
done

labels="supervised"

# supervised runs
for seed in "${seeds[@]}"
do
python proba-train-bert-final.py \
    --train ${indir}/${task}/${task}.train.${labels}.proba.conll.tsv \
    --dev ${indir}/${task}/${task}.dev.${labels}.proba.conll.tsv \
    --test ${indir}/${task}/${task}.test.${labels}.proba.conll.tsv \
    --model ${model} \
    --device cuda \
    --n_epochs ${n_epochs} \
    --lr ${lr} \
    --ignore_masks \
    --seed ${seed} > ${task}.${labels}.seed_${seed}.log
done



