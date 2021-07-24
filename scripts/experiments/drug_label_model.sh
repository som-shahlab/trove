#!/bin/bash
# Train Label Model
#
# Usage:
# -i <input_dir>
# -o <output_dir>
# -n <n_model_search> Number of models for random search
# -m <tier> supervision level
# -t <task>
#
# Example:
# ./drug_label_model.sh \
# -i /Users/fries/Desktop/output/i2b2meds/ \
# -o /Users/fries/Desktop/output/results/ \
# -n 100 \
# -m 1234 \
# -t drug

# command line args
while getopts i:o:n:t:m: flag
do
    case "${flag}" in
        i) indir=${OPTARG};;
        o) outdir=${OPTARG};;
        n) n_model_search=${OPTARG};;
        t) task=${OPTARG};;
        m) tier=${OPTARG};;
    esac
done
echo "input: $indir";
echo "output: $outdir";
echo "n_model_search: $n_model_search";
echo "task: $task";
echo "tier: $tier";

# make output directory if it doesn't exist
mkdir -p ${outdir}

# seeds used for the NC paper
# seeds=(1234 8888 2020 1701 9090 24 126 316 450 541)
# evaluate all LF sets
# top_k=(0 1 2 3 4 5 6 7 8 9 19 49 100)

seeds=(1234)
top_k=(2)

for k in "${top_k[@]}"
do
  for seed in "${seeds[@]}"
  do
    echo "Running k=${k} seed=${seed} | ${indir}/${tier}"
	  python ../../train-label-model.py \
	  --indir ${indir}/${tier}/k${k} \
      --outdir ${outdir}/${task}/${tier}/${k} \
	  --train 0 \
	  --dev 1 \
	  --test 2 \
	  --n_model_search ${n_model_search} \
	  --seed ${seed} > ${outdir}/${task}.k${k}.tier${tier}.seed${seed}.log
	done
done
