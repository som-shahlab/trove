#!/bin/bash
# Generate Proba CoNLL Files
#
# Usage:
# ./drug_proba_conll.sh \
# -m <label_model_checkpoint> \
# -i <label_matrices> \
# -o <outdir> \
# -t <train_json>  \
# -d <dev_json>  \
# -T <test_json>  \
# -e <entity_type>
#
# Example:
# ./drug_proba_conll.sh \
# -m /Users/fries/Desktop/output/results/drug/1234/2/model.n118190.1627096382.pkl \
# -i /Users/fries/Desktop/output/i2b2meds/1234/k2/ \
# -o /Users/fries/Desktop/output/i2b2meds/1234/k2/ \
# -t /Users/fries/Desktop/NER-Datasets-ALL/ehr/i2b2meds/experiments/train.i2b2meds.drug.json  \
# -d /Users/fries/Desktop/NER-Datasets-ALL/ehr/i2b2meds/experiments/dev.i2b2meds.drug.json  \
# -T /Users/fries/Desktop/NER-Datasets-ALL/ehr/i2b2meds/experiments/test.i2b2meds.drug.json  \
# -e drug

# command line args
while getopts i:o:m:e:t:d:T: flag
do
    case "${flag}" in
        i) indir=${OPTARG};;
        o) outdir=${OPTARG};;
        m) model=${OPTARG};;
        e) etype=${OPTARG};;
        t) train=${OPTARG};;
        d) dev=${OPTARG};;
        T) test=${OPTARG};;
    esac
done
echo "input: ${indir}";
echo "output: ${outdir}";
echo "entity type: ${etype}";
echo "model weights: ${model}";
echo "train: ${train}";
echo "dev: ${dev}";
echo "test: ${test}";

# make output directory if it doesn't exist
mkdir -p ${outdir}

python ../../label-model-proba-conll.py \
    --model "${model}" \
    --train "${train}"  \
    --dev "${dev}"  \
    --test "${test}"  \
    --indir "${indir}" \
    --outdir "${outdir}" \
    --etype "${etype}" > ${outdir}/${task}.${labels}.seed_${seed}.log

