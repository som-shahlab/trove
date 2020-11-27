'''
Dumps PubMed abstracts to a common text file format for bulk preprocessing

FORMAT:
~~_PMID_XXXXXX_~~
TEXT
..

We do this in order to use some clunky external tooks for tokenization.

'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import glob
import logging
import argparse
import lxml.etree as et


logger = logging.getLogger(__name__)

def parse_xml_format(filename, outputdir, pmids=None):
    """
    NLM XML Format
    :param filename:
    :param outputdir:
    :return:
    """
    doc_xpath      = './/PubmedArticle'
    id_xpath       = './MedlineCitation/PMID/text()'
    title_xpath    = './MedlineCitation/Article/ArticleTitle/text()'
    abstract_xpath = './MedlineCitation/Article/Abstract/AbstractText/text()'

    n_errs, n_docs = 0, 0
    outfile = os.path.basename(filename)
    outfile = ".".join(outfile.split(".")[0:-1])
    outfile = "{}/{}.txt".format(outputdir.replace(os.path.basename(filename), ""), outfile)

    N = 0
    with open(outfile, "w") as op:
        for i, doc in enumerate(et.parse(filename).xpath(doc_xpath)):
            N += 1
            try:
                pmid = doc.xpath(id_xpath)[0]

                if pmids and pmid not in pmids:
                    continue

                title = doc.xpath(title_xpath)[0] if doc.xpath(title_xpath) else ""
                abstract = doc.xpath(abstract_xpath)[0] if doc.xpath(abstract_xpath) else ""
                text = u"{} {}".format(title, abstract)
                op.write(u"~~_PMID_{}_~~\n".format(pmid))
                op.write(text + u"\n")
                n_docs += 1

            except:
                n_errs += 1
    print(f"Doc N: {N}")
    logger.info("Wrote {}".format(outfile))
    return n_docs, n_errs


def parse_tsv_format(filename, outputdir):
    """

    :param filename:
    :param outputdir:
    :return:
    """

    def load_docs(fpath):
        docs = {}
        with open(fpath, "r") as fp:
            for i, row in enumerate(fp):
                try:
                    row = row.split("\t")
                    if len(row) == 4:
                        pid, title, body, mesh = row
                        text = (title + " " + body)
                    elif len(row) == 3:
                        pid, title, body = row
                        text = (title + " " + body)
                    elif len(row) == 2:
                        pid, text = row
                    docs[pid] = text
                except:
                    print('error', len(row))
        return docs

    docs =  load_docs(filename)

    errors = 0
    outfile = os.path.basename(filename)
    outfile = ".".join(outfile.split(".")[0:-1])
    outfile = "{}/{}.txt".format(outputdir.replace(os.path.basename(filename), ""), outfile)

    with codecs.open(outfile, "w", "utf-8") as op:
        for doc_name, text in docs.items():
            op.write(u"~~_PMID_{}_~~\n".format(doc_name))
            op.write(text + u"\n")

    logger.info("Wrote {}".format(outfile))
    return errors


def get_doc_parser(format):
    """
    Support various utililities for extracting text data

    :param format:
    :return:
    """
    if format == "xml":
        return parse_xml_format
    elif format == "tsv":
        return parse_tsv_format


def main(args):

    doc_parser = get_doc_parser(args.format)

    pmids = open(args.pmids, 'r').read().splitlines() if args.pmids else {}

    filelist = glob.glob("{}/*.xml.gz".format(args.inputdir)) if os.path.isdir(args.inputdir) else [args.inputdir]
    filelist = [fp for fp in filelist if not os.path.isdir(fp)]
    print(filelist)

    N = 0
    for fp in filelist:
        if not os.path.exists(args.outputdir):
            os.mkdir(args.outputdir)

        n_docs,n_errs = doc_parser(fp, args.outputdir, pmids)
        print(n_docs,'\t', fp.split('/')[-1])
        N += n_docs
        if n_errs:
            logger.info("Errors: {}".format(n_errs))
    print(f'Parsed N={N} PMIDS={len(pmids)} {N/len(pmids)*100:2.1f}%')

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument("-i", "--inputdir", type=str,
                           default="input directory or file")
    argparser.add_argument("-o", "--outputdir", type=str,
                           default="outout directory")
    argparser.add_argument("-t", "--pmids", type=str,
                           default="target PMID set")

    argparser.add_argument("-f", "--format", type=str, default="pubtator")
    argparser.add_argument("-p", "--prefix", type=str, default="fixes", help="prefix")
    args = argparser.parse_args()

    args.outputdir = args.outputdir + "/tmp/"

    main(args)
