#!/bin/bash
# Apply Labeling Functions
#
# Usage:
# ./drug_lfs.sh \
# -i <input_dir>
# -o <output_dir>
# -d <data_root> supervision sources
#
# Example:
# ./drug_lfs.sh \
# -i /Users/fries/Desktop/NER-Datasets-ALL/ \
# -o /Users/fries/Desktop/output/i2b2meds/ \
# -d /Users/fries/Code/refactor/inkNER/data/supervision/

# command line args
while getopts i:o:d: flag
do
    case "${flag}" in
        i) indir=${OPTARG};;
        o) outdir=${OPTARG};;
        d) dataroot=${OPTARG};;
    esac
done
echo "Input: $indir";
echo "Output: $outdir";
echo "Data Root: $dataroot";

# make output directory if it doesn't exist
mkdir -p ${outdir}

n_procs=4
unlabeled="${indir}/ehr/i2b2meds/unlabeled/"

task="drug"
corpus="i2b2meds"
domain="ehr"
tag_fmt="BIO"

#top_k=(0 1 2 3 4 5 6 7 8 9 19 49 100)

# best for tier 1234
top_k=(2)
tiers=(1234)

# best for tier 123
#top_k=(1)
#tiers=(123)

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
  --n_procs ${n_procs} > ${outdir}/${task}.${corpus}.${domain}.${tag_fmt}.${k}.${tier}.log
done
done
