#
# SHC notes to TSV format
# Usage:
#    python shc_notes_to_tsv.py \
#    -i /data5/stride8/note_extraction/notes_extracted_V2/
#    -o /data5/stride8_parsed_notes/
#    -b 100000
#

import re
import glob
import argparse


def dataloader(filelist):
    delim = r'''^\[\[ID=([0-9]+)\]\]\t'''
    for fpath in filelist:
        with open(fpath, 'r') as fp:
            curr_id, curr = None, []
            for line in fp:

                m = re.search(delim, line, re.I)
                if m:
                    if curr:
                        yield (curr_id, ''.join(curr))
                    curr_id = m.group(1)
                    curr = [re.sub(delim, '', line)]
                else:
                    curr.append(line)
            if curr:
                yield (curr_id, ''.join(curr))

def escape(text):
    return text.replace('\n', '\\n').replace('\t', '\\t')

def write_tsv(data, fpath):
    with open(fpath, 'w') as fp:
        for note_id, text in data.items():
            fp.write('\t'.join([note_id, escape(text)]) + '\n')
    print(f'Wrote {len(data)} rows to {fpath}')

def main(args):

    note_types = ['radiology', 'clinical', 'pathology']

    block_i = 0
    for name in note_types:
        block = {}
        filelist = glob.glob(f'{args.inputdir}/{name}/notes-{name}*')
        print(name, len(filelist))
        for note_id, text in dataloader(filelist):
            block[note_id] = text
            if len(block) >= args.block_size:
                write_tsv(block, f'{args.outputdir}/{name}.{block_i}.tsv')
                block = {}
                block_i += 1
        if block:
            write_tsv(block, f'{args.outputdir}/{name}.{block_i}.tsv')
            block_i += 1

        print(f'{name} DONE')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-i", "--inputdir", type=str, default=None,
                           required=True, help="input directory")
    argparser.add_argument("-o", "--outputdir", type=str, default=None,
                           required=True, help="output directory")
    argparser.add_argument("-b", "--block_size", type=int, default=1000000,
                           help="size of output TSV blocks")
    args = argparser.parse_args()

    main(args)

# 32 cores d01
# radiology main took: 2543.4726 sec (0.7 hours)
# clinical main took: 30449.9497 sec (8.5 hours)
# pathology main took: 2318.8526 sec (0.7 hours)

# python shc_notes_to_tsv.py -i /data5/stride8/note_extraction/notes_extracted_V2/ -o /data5/stride8_parsed_notes/ -b 1000000

# python /data4/jfries/code/ehr-rwe/preprocessing/parse.py \
# --inputdir /data5/stride8_tsv_notes/radiology/ \
# --outputdir /data5/stride8_nlp_notes/radiology/ \
# --prefix shc-radiology \
# --n_procs 32 \
# --disable ner,parser,tagger,lemmatizer \
# --batch_size 250000 \
# --keep_whitespace \
# --no_header


# python /data4/jfries/code/ehr-rwe/preprocessing/parse.py \
# # --inputdir /data5/stride8_tsv_notes/clinical/ \
# # --outputdir /data5/stride8_nlp_notes/clinical/ \
# # --prefix shc-clinical \
# # --n_procs 32 \
# # --disable ner,parser,tagger,lemmatizer \
# # --batch_size 250000 \
# # --keep_whitespace \
# # --no_header

#
# python /data4/jfries/code/ehr-rwe/preprocessing/parse.py \
# --inputdir /data5/stride8_tsv_notes/pathology/ \
# --outputdir /data5/stride8_nlp_notes/pathology/ \
# --prefix shc-pathology \
# --n_procs 16 \
# --disable ner,parser,tagger,lemmatizer \
# --batch_size 250000 \
# --keep_whitespace \
# --no_header
#
#
# dump-sentences.py /data5/stride8_nlp_notes/radiology/ > stride8_nlp_notes/radiology.sentences.txt
# dump-sentences.py /data5/stride8_nlp_notes/clinical/ > stride8_nlp_notes/clinical.sentences.txt
#python dump-sentences.py /data5/stride8_nlp_notes/pathology/ > stride8_nlp_notes/pathology.sentences.txt
