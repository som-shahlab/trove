"""

python raw_note_export.py -i <INPUT> -o <OUTPUT> -b 1000

"""
import os
import re
import sys
import glob
import argparse
from toolz import partition_all


def mimic_preprocessing(text):
    """

    :param text:
    :return:
    """
    # remove junk headers that concatenate multiple notes
    sents = []
    skip = False
    for line in text.split('\n'):
        if line.strip() == '(Over)':
            skip = True
        elif line.strip() == '(Cont)':
            skip = False
            continue
        if not skip:
            sents.append(line)
    text = '\n'.join(sents)

    return text


def mimic_doc_preprocessor(s):
    """
    Remove MIMIC-III anonymization tags of the form
    [**First Name8 (NamePattern2) **]
    otherwise tokenization breaks.
    """
    rgx = r'''(\[\*\*)[a-zA-Z0-9_/()\- ]+?(\*\*\])'''
    for m in re.finditer(rgx, s):
        repl = m.group().replace('[**', '|||').replace('**]', '|||')
        repl = re.sub("[/()]", "|", repl)
        s = s.replace(m.group(), repl.replace(" ", "_"))

    m = re.search(r'''[?]{3,}''', s)
    if m:
        s = s.replace(m.group(), u"•" * (len(m.group())))
    return s


def save_tsv(data, outfpath):
    with open(outfpath, 'w') as fp:
        fp.write("DOC_NAME\tTEXT\n")
        for row in data:
            row = '\t'.join(row)
            fp.write(f"{row}\n")


def main(args):

    filelist = glob.glob(f'{args.inputdir}/*')
    batches = partition_all(args.batch_size, filelist)
    print(f'Documents: {len(filelist)}')

    if not os.path.exists(args.outputdir):
        print("created output directory")
        os.mkdir(args.outputdir)

    for i,batch in enumerate(batches):
        data = []
        for fpath in batch:
            doc_name = fpath.split("/")[-1].split(".")[0]
            text = open(fpath,'r').read()
            if args.fmt == 'mimic':
                # text = mimic_preprocessing(text) \
                #     if args.preprocess == 'mimic' else text
                text = mimic_doc_preprocessor(text) \
                    if args.preprocess == 'mimic' else text

            # escape whitespace
            text = text.replace('\n', '\\n').replace('\t', '\\t')
            data.append((doc_name, text))

        outfpath = f'{args.outputdir}/{args.batch_size}.{i}.tsv'
        print(outfpath)
        save_tsv(data, outfpath)
        data = []

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument("-i", "--inputdir", type=str, default=None,  help="input directory")
    argparser.add_argument("-o", "--outputdir", type=str, default=None, help="output directory")
    argparser.add_argument("-b", "--batch_size", type=int, default=1000, help="batch size")
    argparser.add_argument("-f", "--fmt", type=str, default="mimic", help="document source format")
    argparser.add_argument("-e", "--export_fmt", type=str, default="tsv", help="document export format")
    argparser.add_argument("-P", "--preprocess", type=str, default=None, help="preprocess docs")
    args = argparser.parse_args()

    main(args)