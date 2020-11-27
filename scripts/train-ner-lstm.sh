python train-lstm.py \
--train /Users/fries/Desktop/NER-Datasets-ALL/pubmed/cdr/conll/train.cdr-disease.txt \
--dev /Users/fries/Desktop/NER-Datasets-ALL/pubmed/cdr/conll/dev.cdr-disease.txt \
--test /Users/fries/Desktop/NER-Datasets-ALL/pubmed/cdr/conll/test.cdr-disease.txt \
--input_emb_dim 50 \
--lstm_num_layers 1 \
--lr 0.005 \
--n_epochs 25