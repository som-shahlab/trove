"""
python train-bert-wordpiece-tokenizer.py \
--files /Users/fries/medline_pubmed.sentences.raw.txt \
--out /Users/fries/pubmed-wordpiece/ \
--name bert-pubmed-wordpiece-cased

"""
import sys
import glob
import argparse
from tokenizers import BertWordPieceTokenizer


def main(args):

    tokenizer = BertWordPieceTokenizer(
        clean_text=True, handle_chinese_chars=True, strip_accents=True,
        lowercase=False,
    )

    trainer = tokenizer.train(
        files,
        vocab_size=args.vocab_size,
        min_frequency=2,
        show_progress=True,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
        limit_alphabet=1000,
        wordpieces_prefix="##",
    )

    # Save the files
    tokenizer.save(args.out, args.name)

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--files",
        default=None,
        metavar="path",
        type=str,
        required=True,
        help="The files to use as training; accept '**/*.txt' \
        type of patterns if enclosed in quotes",
    )
    parser.add_argument(
        "--out",
        default="./",
        type=str,
        help="Path to the output directory, where the files will be saved",
    )
    parser.add_argument(
        "--name",
        default="bert-wordpiece",
        type=str,
        help="The name of the output vocab files"
    )

    parser.add_argument("--vocab_size", type=int, default=32000)
    args = parser.parse_args()

    files = glob.glob(args.files)
    if not files:
        print(f"File does not exist: {args.files}")
        sys.exit(1)

    main(args)

# python train-bert-wordpiece-tokenizer.py \
# --files shc.sentences.2020_04_04.txt \
# --out /data5/stride8_nlp_notes/wordpiece/