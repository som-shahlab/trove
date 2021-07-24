#!/bin/bash
# Train BioBERT End Model
#
# Usage:
#
# Example:
# ./drug_end_model.sh \
# -m /Users/fries/Desktop/trove-final/biobert_v1.1_pubmed-pytorch \
# -t /Users/fries/Desktop/output/i2b2meds/1234/k2/drug.train.prior_balance.proba.conll.tsv \
# -d /Users/fries/Desktop/output/i2b2meds/1234/k2/drug.dev.prior_balance.proba.conll.tsv \
# -T /Users/fries/Desktop/output/i2b2meds/1234/k2/drug.test.prior_balance.proba.conll.tsv \
# -l 1e-5 \
# -e 50

# command line args
while getopts m:t:d:T:l:e: flag
do
    case "${flag}" in
        m) model=${OPTARG};;
        t) train=${OPTARG};;
        d) dev=${OPTARG};;
        T) test=${OPTARG};;
        l) lr=${OPTARG};;
        e) n_epochs=${OPTARG};;
    esac
done

echo "model weights: ${model}";
echo "train: ${train}";
echo "dev: ${dev}";
echo "test: ${test}";
echo "lr: ${lr}";
echo "n_epochs: ${n_epochs}";

device="cpu"
cuda_visible_devices="" #"CUDA_VISIBLE_DEVICES=0 "

seeds=(1234)

# weakly supervised runs
for seed in "${seeds[@]}"
do
${cuda_visible_devices} python ../../proba-train-bert.py \
    --train ${train} \
    --dev ${dev} \
    --test ${test} \
    --model ${model} \
    --device ${device} \
    --n_epochs ${n_epochs} \
    --lr ${lr} \
    --seed ${seed} > bert.seed_${seed}.log
done

# labels="supervised"
#
# # supervised runs
# for seed in "${seeds[@]}"
# do
# python proba-train-bert-final.py \
#     --train ${indir}/${task}/${task}.train.${labels}.proba.conll.tsv \
#     --dev ${indir}/${task}/${task}.dev.${labels}.proba.conll.tsv \
#     --test ${indir}/${task}/${task}.test.${labels}.proba.conll.tsv \
#     --model ${model} \
#     --device cuda \
#     --n_epochs ${n_epochs} \
#     --lr ${lr} \
#     --ignore_masks \
#     --seed ${seed} > ${task}.${labels}.seed_${seed}.log
# done
