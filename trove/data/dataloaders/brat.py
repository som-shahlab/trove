import os
import re
import sys
import glob
import copy
import itertools
import numpy as np
from collections import defaultdict

###################################################################
#
# BRAT Objects
#
###################################################################

class BratBase(object):
    """
    Once initalized, all BRAT objects are treated as immutable for
    computing hashes.
    """

    def __init__(self, id_, type_, doc_name, cid=None):
        self.id_ = id_
        self.cid = cid # canonical id for entity linking
        self.type_ = type_
        self.doc_name = doc_name
        self.attribs = {}
        self.notes = {}
        self.symbol = self.__class__.__name__.upper()

    def clone(self, ignore_attributes=False):
        item = copy.deepcopy(self)
        if ignore_attributes:
            item.attribs = {}
        return item

    def attribute(self, key):
        return self.attribs[key] if key in self.attribs else False

    @property
    def type(self):
        return self.type_

    def __gt__(self, other):
        return other.type_ < self.type_

    def __eq__(self, other):
        return False if self.__hash__() != other.__hash__() else True

    def __repr__(self):
        return "<brat.{}>".format(self.__class__.__name__)


class Entity(BratBase):
    def __init__(self, id_, doc_name, entity_type, span, text):
        super(Entity, self).__init__(id_, entity_type, doc_name)
        self.span = span
        self.text = text
        # for now, assume continuous entities
        self.abs_char_start = span[0][0]
        self.abs_char_end = span[0][-1]

    def get_stable_id(self):
        return (self.type_, self.doc_name, self.span, tuple(self.attribs.keys()))

    def __hash__(self):
        """These values are assumed to be immutable (otherwise hashing breaks)"""
        v = (self.type_, self.doc_name, self.span, self.text, tuple(self.attribs.keys()))
        return hash(v)

    def __str__(self):
        s = ",".join(["{}:{}".format(*s) for s in self.span])
        attribs = "[{}]".format(",".join(self.attribs.keys())) if self.attribs else ""
        return "{}{}(\"{}\":{}-{})".format(self.type_, attribs, self.text, self.abs_char_start, self.abs_char_end)

    def __repr__(self):
        return self.__str__()


class Relation(BratBase):
    def __init__(self, id_, doc_name, rela_type, args):
        super(Relation, self).__init__(id_, rela_type, doc_name)
        self.args = args
        self.abs_char_start = 0
        self.abs_char_end = 0

    def init_args(self, entity_map):
        self.args = sorted([entity_map[arg_id] for arg_id in self.args], key=lambda x: x.type_, reverse=1)
        self.abs_char_start = self.args[0].abs_char_start
        self.abs_char_end = self.args[-1].abs_char_end

    def get_stable_id(self):
        args = tuple(sorted([arg.get_stable_id() for arg in self.args]))
        return (self.type_, self.doc_name, args)

    def clone(self, ignore_attributes=False):
        # clone Entity args
        arg1 = self.args[0].clone(ignore_attributes)
        arg2 = self.args[1].clone(ignore_attributes)
        # create cloned relation
        c = Relation(self.id_, self.doc_name, self.type_, [arg1, arg2])
        c.abs_char_start = self.args[0].abs_char_start
        c.abs_char_end = self.args[-1].abs_char_end
        return c

    def __getitem__(self, key):
        arg_types = [entity.type for entity in self.args]
        assert key in arg_types
        return self.args[arg_types.index(key)]

    def __hash__(self):
        v = (self.type_, self.doc_name, tuple(self.args))
        return hash(v)

    def __str__(self):
        args = [str(arg) for arg in self.args]
        return "{}( {} )".format(self.type_, ", ".join(args))


class Event(BratBase):
    def __init__(self, id_, doc_name, evt_type, args):
        super(Event, self).__init__(id_, evt_type, doc_name)
        self.args = args

    def init_args(self, entity_map):
        self.args = sorted([entity_map[arg_id] for arg_id in self.args], key=lambda x: x.type_, reverse=1)

    def get_stable_id(self):
        args = tuple(sorted([arg.get_stable_id() for arg in self.args]))
        return (self.type_, self.doc_name, args, tuple(self.attribs.keys()))

    def __str__(self):
        args = [str(arg) for arg in self.args]
        return "{}( {} )".format(self.type_, ", ".join(args))


###################################################################
#
# Standoff & BRAT Config Parser
#
###################################################################

class StandoffParser(object):
    """
    Standoff Annotation Parser

    See:
        BioNLP Shared Task 2011     http://2011.bionlp-st.org/home/file-formats
        Brat Rapid Annotation Tool  http://brat.nlplab.org/standoff.html

    Annotation ID Types
    T: text-bound annotation
    R: relation
    E: event
    A: attribute
    M: modification (alias for attribute, for backward compatibility)
    N: normalization [new in v1.3]
    #: note

    Many of of the advanced schema abilities used by BRAT are not implemented, so
    mind the following caveats:

    (1) We do not currently support hierarchical entity definitions, e.g.,
            !Anatomical_entity
                !Anatomical_structure
                    Organism_subdivision
                    Anatomical_system
                    Organ
    (2) All relations must be binary with a single argument type
    (3) Attributes, normalization, and notes are added as candidate meta information

    """

    TEXT_BOUND_ID = 'T'
    RELATION_ID = 'R'
    EVENT_ID = 'E'
    ATTRIB_ID = 'A'
    MOD_ID = 'M'
    NORM_ID = 'N'
    NOTE_ID = '#'

    SYMBOLS = {"T": "Entity", "R": "Relation", "E": "Event"}

    def __init__(self, encoding="utf-8"):
        """
        Initialize standoff annotation parser

        :param encoding:
        """
        self.encoding = encoding

    def load_annotations(self, input_dir):
        """
        Import BART project,
        :param input_dir:
        :param autoreload:
        :param num_threads:
        :param parser:
        :return:
        """
        config_path = "{}/{}".format(input_dir, "annotation.conf")
        if not os.path.exists(config_path):
            print("Fatal error: missing 'annotation.conf' file", file=sys.stderr)
            return

        # load brat config (this defines relation and argument types)
        config = self._parse_config(config_path)
        anno_filelist = set([os.path.basename(fn)[:-4] for fn in glob.glob(input_dir + "/*.ann")])

        # import standoff annotations for all documents
        annotations = {}
        for doc_name in anno_filelist:
            txt_fn = "{}/{}.txt".format(input_dir, doc_name)
            ann_fn = "{}/{}.ann".format(input_dir, doc_name)
            if os.path.exists(txt_fn) and os.path.exists(ann_fn):
                annotations[doc_name] = self._parse_annotations(txt_fn, ann_fn)

        return annotations

    def _parse_annotations(self, txt_filename, ann_filename):
        """
        Use parser to import BRAT standoff format

        :param txt_filename:
        :param ann_filename:
        :return:
        """
        annotations, attributes = {}, {}
        doc_name = txt_filename.split("/")[-1].split(".")[0]

        # read document string
        with open(txt_filename, "rU") as fp:
            doc_str = fp.read()

        # load annotations
        with open(ann_filename, "rU") as fp:
            for line in fp:
                if not line.strip():
                    continue
                try:
                    row = line.strip().split("\t")
                    anno_id_prefix = row[0][0]
                except:
                    print(line)
                    print(row)

                # parse each entity/relation type
                if anno_id_prefix == StandoffParser.TEXT_BOUND_ID:
                    anno_id, entity, text = row
                    entity_type = entity.split()[0]
                    spans = [list(map(int, x.split())) for x in entity.lstrip(entity_type).split(";")]
                    spans = tuple([tuple(s) for s in spans])
                    
                    # santity check to see if label span matches document span
                    text = text.strip()
                    mention = " ".join([doc_str[i:j] for i,j in spans]).replace("\n", " ").strip()
                    if mention != text:
                        err_msg = f"Error spans do not match: {doc_name}:{spans} mention('{mention}') != text('{text}')"
                        print(err_msg, file=sys.stderr)
                        #continue

                    annotations[anno_id] = Entity(anno_id, doc_name, entity_type, spans, mention)
                    
                elif anno_id_prefix in [StandoffParser.RELATION_ID, '*']:
                    # format: Complication Arg1:T65 Arg2:T74
                    anno_id, rela = row
                    rela_type, arg1, arg2 = rela.split()
                    arg1 = arg1.split(":")[-1] if ":" in arg1 else arg1
                    arg2 = arg2.split(":")[-1] if ":" in arg2 else arg2
                    annotations[anno_id] = Relation(anno_id, doc_name, rela_type, (arg1, arg2))

                elif anno_id_prefix == StandoffParser.EVENT_ID:
                    # FORMAT: <ANNO_ID>\t<TYPE:ARG_ID>*
                    # example: E1	Pain:T53 Anatomy:T59 Severity:T60
                    # where first argument type is Event type
                    anno_id, args = row
                    args = [arg.split(":") for arg in args.split()]
                    types, args = zip(*args)
                    annotations[anno_id] = Event(anno_id, doc_name, types[0], args)

                elif anno_id_prefix == StandoffParser.ATTRIB_ID:
                    anno_id, attrib = row
                    attrib, arg_id = attrib.split()
                    attributes[anno_id] = (attrib, arg_id)

        # Initalize Relation, Event dependencies
        for anno_id in annotations:
            if anno_id[0] not in ['R', 'E']:
                continue
            # initialize child entities
            annotations[anno_id].init_args(annotations)

        # Assign Attributes
        for anno_id in attributes:
            attrib, arg_id = attributes[anno_id]
            annotations[arg_id].attribs[attrib] = True

        return list(annotations.values())

    def _normalize_relation_name(self, name):
        """
        Normalize relation name

        :param name:
        :return:
        """
        name = re.split("[-_]", name)
        if len(name) == 1:
            return name[0]
        name = [x.lower() for x in name]
        return "".join([x[0].upper() + x[1:] for x in name])

    def _parse_config(self, filename):
        """
        Parse BRAT annotation.config

        :param filename:
        :return:
        """
        config = defaultdict(list)
        with open(filename, "rU") as fp:
            curr = None
            for line in fp:
                # skip comments
                line = line.strip()
                if not line or line[0] == '#':
                    continue
                # brat definition?
                m = re.search("^\[(.+)\]$", line)
                if m:
                    curr = m.group(1)
                    continue
                config[curr].append(line)

        # type-specific parsing
        tmp = []
        for item in config['relations']:
            m = re.search("^(.+)\s+Arg1:(.+),\s*Arg2:(.+),*\s*(.+)*$", item)
            name, arg1, arg2 = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
            # convert relations to camel case
            name = self._normalize_relation_name(name)
            arg2 = arg2.split(",")[0]  # strip any <rel-type> defs
            arg1 = arg1.split("|")
            arg2 = arg2.split("|")
            tmp.append((name, arg1, arg2))
        config['relations'] = tmp

        tmp = []
        for item in config['attributes']:
            name, arg = item.split()
            arg = arg.split(":")[-1]
            tmp.append((name, arg))
        config['attributes'] = tmp

        return config


###################################################################
#
# BRAT Annotation Interface
#
###################################################################

class BratAnnotations(object):
    def __init__(self, inputdir, encoding="utf-8"):
        self.encoding = encoding
        self.standoff_parser = StandoffParser(encoding=self.encoding)

        # load annotation sets (these must be the same doc sets, but different annotators)
        self.annotations = [fp for fp in glob.glob("{}/*".format(inputdir)) if os.path.isdir(fp)]
        self.annotations = {fpath.split("/")[-1]: fpath for fpath in self.annotations}

        for annotator, fpath in self.annotations.items():
            self.annotations[annotator] = self.load_annotations(fpath)

    def load_annotations(self, fpath):
        return self.standoff_parser.load_annotations(fpath)

    def get_doc_names(self):
        return set(
            itertools.chain.from_iterable([self.annotations[annotator].keys() for annotator in self.annotations]))

    def annotator_agreement(self, ignore_types=[], relations_only=False, method="randolph"):
        """
        Compute agreement measures

        :param ignore_types:
        :param relations_only:
        :return:
        """
        annos = self.aggregate_raters(ignore_types, relations_only)

        # overall score
        V = np.vstack(annos.values())
        kappa = fleiss_kappa(V, method=method)
        print("Kappa Agreement (method={})".format(method))
        print("{:<14} {:>6}: {:.3f}".format('OVERALL', "(n={})".format(len(V)), kappa))

        # break down by annotation type
        types = np.unique([anno.type for anno in annos])
        for t in [t for t in types if t not in ignore_types]:
            v = np.vstack([annos[anno] for anno in annos if anno.type == t])
            kappa = fleiss_kappa(v, method=method)
            print("{:<14} {:>6}: {:.3f}".format(t, "(n={})".format(len(v)), kappa))

    def aggregate_raters(self, ignore_types=[], relations_only=False):
        """
        Build a matrix of label agreements across multiple annotators. This assumes:
         1) a fixed set of annotators *per document*
         2) absence of an annotation within document is equivalent to a vote of False

        return n x 2 table of vote counts   | True | False |

        :param ignore_types:
        :param relations_only:
        :return:
        """
        # sort annotations by document
        annos_by_doc = defaultdict(dict)
        annotators_by_anno = defaultdict(dict)
        annotators_by_doc = defaultdict(dict)

        for annotator in self.annotations:
            for doc_name in self.annotations[annotator]:
                annotators_by_doc[doc_name][annotator] = 1
                doc_annos = self.annotations[annotator][doc_name]
                doc_annos = self._filter_annotations(doc_annos, ignore_types, relations_only)
                for anno in doc_annos:
                    annos_by_doc[doc_name][anno] = 1
                    annotators_by_anno[anno][annotator] = 1

        # build annotation matrix by document
        # absence of an annotation for a given annotator implies False
        M = defaultdict(dict)
        for doc_name in annos_by_doc:
            annotators = set(annotators_by_doc[doc_name].keys())
            for anno in annos_by_doc[doc_name]:
                M[anno] = [0, 0]
                for name in annotators:
                    if name in annotators_by_anno[anno]:
                        M[anno][0] += 1
                    else:
                        M[anno][1] += 1

        # sanity check
        V = np.vstack(M.values())
        sum_annotators = np.sum(V, axis=1)
        
        if np.unique(sum_annotators).shape[0] > 1:
            msg = "WARNING! different number of annotators per document detected:\n"
            errors = {anno.doc_name for anno in M if np.sum(M[anno]) < max(sum_annotators)}
            for anno in M:
                if np.sum(M[anno]) < max(sum_annotators):
                    msg += " {} {}\n".format(anno.doc_name, M[anno])
            sys.stderr.write(msg)

        return M

    def _filter_annotations(self, items, ignore_types, relations_only=False):
        """
        """
        # filter BRAT entities
        doc_annotations = [anno for anno in items if anno.type not in ignore_types]
        # filter to only entities found in Relations
        if relations_only:
            relations = [anno for anno in doc_annotations if anno.symbol == "RELATION"]
            entities = list(itertools.chain.from_iterable([rela.args for rela in relations]))
            doc_annotations = list(set(relations + entities))
        return doc_annotations

    def annotator_summary(self):
        print("=" * 50)
        print("Annotator Summary".upper())
        print("=" * 50)

        for annotator in self.annotations:
            print("-" * 25)
            print(annotator)
            print("-" * 25)
            types = defaultdict(lambda: defaultdict(list))

            for doc_name in self.annotations[annotator]:
                for anno in self.annotations[annotator][doc_name]:
                    class_name = anno.__class__.__name__
                    types[class_name][anno.type_].append(anno)

            for cls in sorted(types):
                n = sum([len(types[cls][subtype]) for subtype in types[cls]])
                print("{:<18}  {}".format(cls.upper(), n))
                for subtype in sorted(types[cls]):
                    print(" - {:<15}  {}".format(subtype, len(types[cls][subtype])))
                print(" " * 25)

    def init_labels(self, class_map, types, adjudication="mv", verbose=True):

        self.labels = {}
        annotations = self.aggregate_raters()

        for anno in annotations:
            if anno.type not in types:
                continue
            if adjudication == "mv":
                if annotations[anno][0] > annotations[anno][1]:
                    self.labels[anno] = class_map(anno)
            elif adjudication == "unanimous":
                if annotations[anno][0] > 1 and annotations[anno][1] == 0:
                    self.labels[anno] = class_map(anno)

        if verbose:
            for t in types:
                l = [1 for anno in self.labels if anno.type == t]
                print("{}: n={}".format(t, len(l)))

                
    def get_ooc(self, candidates, ignore_attributes=True):
        brat_cands = snorkel_to_brat(candidates, ignore_attributes)
        if ignore_attributes:
            gold = {c.clone(ignore_attributes=True): self.labels[c] for c in self.labels}
        else:
            gold = self.labels
        return [c for c in gold if c not in brat_cands]
        
    def get_labels(self, candidates, ignore_attributes=True, neg_label=0):
        """
        For a set of Snorkel candidates, create and array of Candidates.
        """
        #brat_cands = snorkel_to_brat(candidates, ignore_attributes)
        if ignore_attributes:
            gold = {c.clone(ignore_attributes=True): self.labels[c] for c in self.labels}
        else:
            gold = self.labels
        return np.array([1 if c in gold and gold[c] == 1 else neg_label for c in candidates])

    def score(self, candidates, brat_candidates, pred_labels, ignore_attributes=True, neg_label=0, ooc_correct=True):
        # In some cases, we want to ignore BRAT attributes when computing metrics, e.g.,
        # in binary classification, we don't have enough information provided by the 0-1
        # label to infer the underlying attribute class space (which is 1...n)
        if ignore_attributes:
            gold = {c.clone(ignore_attributes=True): self.labels[c] for c in self.labels}
        else:
            gold = {c: self.labels[c] for c in self.labels}

        # convert to BRAT candidates
        #brat_cands = snorkel_to_brat(candidates, ignore_attributes)
        brat2snorkel = dict(zip(brat_candidates, candidates))
        
        # filter gold cands to only include documents from 'candidates'
        doc_set = set([c[0].get_parent().document.name for c in candidates])
        gold = {c: gold[c] for c in gold if c.doc_name in doc_set}
        
        # error buckets
        bins = defaultdict(list)
        for i, c in enumerate(brat_candidates):
            if c in gold and gold[c] == 1:
                bins['tp' if pred_labels[i] == 1 else 'fn'].append(c)
            elif c in gold and gold[c] == neg_label:
                bins['fp' if pred_labels[i] == 1 else 'tn'].append(c)
            elif c not in gold:
                bins['fp' if pred_labels[i] == 1 else 'tn'].append(c)
            else:
                print("ERROR", pred_labels[i], c in gold, gold[c])
            
        # out-of-candidate items (if we don't account for this, our recall is under-estimated)
        ooc = 0
        brat_candidates = set(brat_candidates)
        for c in gold:
            if c not in brat_candidates and gold[c] == 1:
                ooc += 1

        # convert back to Snorkel candidates
        for bucket in bins:
            bins[bucket] = [brat2snorkel[c] for c in bins[bucket] if c in brat2snorkel]

        ntp, nfp, ntn, nfn = len(bins["tp"]), len(bins["fp"]), len(bins["tn"]), len(bins["fn"])
        if not ooc_correct:
            ooc = 0
        print_scores(ntp, nfp, ntn, nfn, ooc, title='Scores')

        return bins


###################################################################
#
# Misc. Helper Functions
#
###################################################################

def fleiss_kappa(X, method="fleiss"):
    """
    Fleiss' Kappa (Fleiss 1971)

    (from statsmodels module documentation)
     Method 'fleiss' returns Fleiss' kappa which uses the sample margin
        to define the chance outcome.
        Method 'randolph' or 'uniform' (only first 4 letters are needed)
        returns Randolph's (2005) multirater kappa which assumes a uniform
        distribution of the categories to define the chance outcome.
    """
    num_annotators = np.unique(np.sum(X, axis=1))
    assert num_annotators.shape[0] == 1 and num_annotators > 0
    N, k = X.shape
    p = np.sum(X, axis=0) / (N * num_annotators)
    P = (np.sum(X * X, axis=1) - num_annotators) / (num_annotators * (num_annotators - 1))
    p_mean = np.sum(P) / N
    if method == 'fleiss':
        p_mean_exp = np.sum(p * p)
    elif method in ['randolph', 'uniform']:
        p_mean_exp = 1.0 / k
    return (p_mean - p_mean_exp) / (1 - p_mean_exp)


def binary_scores_from_counts(ntp, nfp, ntn, nfn):
    """
    Precision, recall, and F1 scores from counts of TP, FP, TN, FN.
    Example usage:
        p, r, f1 = binary_scores_from_counts(*map(len, error_sets))
    """
    prec = ntp / float(ntp + nfp) if ntp + nfp > 0 else 0.0
    rec = ntp / float(ntp + nfn) if ntp + nfn > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if prec + rec > 0 else 0.0
    return prec, rec, f1


def print_scores(ntp, nfp, ntn, nfn, ooc=0, title='Scores'):
    if ooc > 0:
        nfn += ooc
    prec, rec, f1 = binary_scores_from_counts(ntp, nfp, ntn, nfn)
    pos_acc = ntp / float(ntp + nfn) if ntp + nfn > 0 else 0.0
    neg_acc = ntn / float(ntn + nfp) if ntn + nfp > 0 else 0.0
    print("=============================================")
    print(title)
    print("=============================================")
    print("Pos. class accuracy: {:.3}".format(pos_acc))
    print("Neg. class accuracy: {:.3}".format(neg_acc))
    print("Precision            {:.3}".format(prec))
    print("Recall               {:.3}".format(rec))
    print("F1                   {:.3}".format(f1))
    print("---------------------------------------------")
    if ooc:
        print("TP: {} | FP: {} | TN: {} | FN: {}* (OOC: {})".format(ntp, nfp, ntn, nfn, ooc))
    else:
        print("TP: {} | FP: {} | TN: {} | FN: {}".format(ntp, nfp, ntn, nfn))
    print("=============================================\n")


def snorkel_to_brat(candidates, ignore_attributes=False):
    """
    Convert Snorkel 0.7 candidates to BRAT objects.
    HACK: Hard-coded for Complication relations
    TODO: make general
    """
    cands = []
    for c in candidates:
        entities = {}
        doc_name = c.get_parent().document.name
        abs_start = c.get_parent().abs_char_offsets[0]

        span = ((c.implant.char_start + abs_start, c.implant.char_end + abs_start + 1),)
        entities["T1"] = Entity("T1", doc_name, "Implant", span, c.implant.get_attrib_span("words"))

        span = ((c.complication.char_start + abs_start, c.complication.char_end + abs_start + 1),)
        entities["T2"] = Entity("T2", doc_name, "Finding", span, c.complication.get_attrib_span("words"))

        relation = Relation("R1", doc_name, rela_type="Complication", args=["T1", "T2"])
        relation.init_args(entities)

        relation = relation.clone(ignore_attributes=ignore_attributes)

        cands.append(relation)

    return cands


def doc_to_text(doc):
    """
    Convert document object to original text represention.
    Assumes parser offsets map to original document offsets

    :param doc:
    :param sent_delim:
    :return:
    """
    text = u""
    for i, sent in enumerate(doc.sentences):
        # setup padding so that BRAT displays a minimal amount of newlines
        # while still preserving char offsets
        if len(text) != sent.abs_char_offsets[0]:
            padding = (sent.abs_char_offsets[0] - len(text))
            text += ' ' * (padding - 1) + u"\n"
        text += sent.text.rstrip(u' \t\n\r')
    return text