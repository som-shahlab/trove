from __future__ import annotations

import collections
import csv
import gzip
import io
import os
import re
import tarfile
import urllib.request
import zipfile
from abc import ABCMeta, abstractmethod
from typing import (
    BinaryIO,
    Collection,
    Dict,
    Generic,
    List,
    Literal,
    Mapping,
    Optional,
    Set,
    TypeVar,
    Union,
    cast,
)

from .norm import lowercase
from .stopwords import get_stopwords
from .umls import UMLS

T = TypeVar("T")


class Dictionary(Generic[T], metaclass=ABCMeta):
    def __init__(self, trove_path: str, dictionary_name: str):
        self.trove_path = trove_path
        self.dictionary_name = dictionary_name

        self.full_path = os.path.join(trove_path, dictionary_name)
        self.words: Optional[T] = None

    def get_words(self) -> T:
        if self.words is not None:
            return self.words
        else:
            if not os.path.exists(self.full_path):
                self.download()
            self.words = self.load()
            return self.words

    @abstractmethod
    def get_url(self) -> str:
        ...

    def download(self) -> None:
        opener = urllib.request.build_opener()
        opener.addheaders = [("User-agent", "Mozilla/5.0")]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(self.get_url(), self.full_path)

    @abstractmethod
    def load(self) -> T:
        ...


class ChebiOntology(Dictionary[Collection[str]]):
    def __init__(
        self, trove_path: str, ignore_case: bool, min_tok_len: int = 1
    ):
        super().__init__(trove_path, "ChebiOntology")
        self.ignore_case = ignore_case
        self.min_tok_len = min_tok_len

    def get_url(self) -> str:
        return "ftp://ftp.ebi.ac.uk/pub/databases/chebi/Flat_file_tab_delimited/names.tsv.gz"

    def load(self) -> Collection[str]:
        stop = get_stopwords()

        terms = []
        with gzip.open(self.full_path, "rt") as f:
            reader = csv.DictReader(f, delimiter="\t")

            for line in reader:
                terms.append(line["NAME"])

        terms = [
            lowercase(t) if self.ignore_case else t
            for t in terms
            if t.lower() not in stop
        ]
        terms = [t for t in terms if len(t) >= self.min_tok_len]

        return set(terms)


class AdamDictionary(Dictionary[Dict[str, Set[str]]]):
    def __init__(self, trove_path: str, target_concepts: Collection[str]):
        super().__init__(trove_path, "AdamDictionary")

        self.target_concepts = target_concepts

    def get_url(self) -> str:
        return (
            "http://arrowsmith.psych.uic.edu/arrowsmith_uic/download/adam.tar"
        )

    def load(self) -> Dict[str, Set[str]]:
        synset: Dict[str, Set[str]] = collections.defaultdict(set)
        with tarfile.open(self.full_path) as f:
            possible_adam_file: Optional[BinaryIO] = cast(
                BinaryIO, f.extractfile("adam_database")
            )  # TODO: remove this after upgrading to mypy 0.784
            if possible_adam_file is None:
                raise RuntimeError(
                    "Could not find the adam_database file within the downloaded tar"
                )
            with io.TextIOWrapper(possible_adam_file) as adam_file:
                for i, line in enumerate(adam_file):
                    if line[0] == "#":
                        continue
                    (
                        pref_abbrv,
                        alt_abbrv,
                        long_form,
                        score,
                        num,
                    ) = line.strip().split("\t")

                    long_form = long_form.split(":")[0]
                    alt_abbrv = alt_abbrv.split(":")[0]

                    if float(score) < 0.5:
                        continue

                    if (
                        long_form in self.target_concepts
                        or lowercase(long_form) in self.target_concepts
                    ):
                        synset[pref_abbrv].add(lowercase(long_form))

        return synset


class SpecialistDictionary(Dictionary[Dict[str, Set[str]]]):
    def __init__(
        self,
        trove_path: str,
        umls: UMLS,
        target_concepts: Collection[str],
        filter_ambiguous: Union[bool, float] = True,
    ):
        super().__init__(trove_path, "SpecialistDictionary")

        self.umls = umls
        self.target_concepts = target_concepts
        self.filter_ambiguous = filter_ambiguous

    def get_url(self) -> str:
        return "https://lexsrv3.nlm.nih.gov/LexSysGroup/Projects/lexicon/2020/release/LEX/LRABR"

    def load(self) -> Dict[str, Set[str]]:
        specialist = collections.defaultdict(list)
        with open(self.full_path, "r") as fp:
            for i, line in enumerate(fp):
                row = line.strip().split("|")
                uid, abbrv, atype, uid2, term, _ = row
                if (
                    atype not in ["acronym", "abbreviation"]
                    or not abbrv.isupper()
                ):
                    continue
                # fetch all semantic types linked to this abbreviation and term
                stys = self.umls.get_term_stys(term)
                stys.extend(self.umls.get_term_stys(abbrv))

                ambiguous = False
                if self.filter_ambiguous:
                    # threshold by class probability
                    if type(self.filter_ambiguous) is float:
                        tmp1 = list(zip(*stys))[-1] if stys else []
                        tmp = {sty: tmp1.count(sty) for sty in tmp1}
                        wsd = {True: 0, False: 0}
                        for sty in tmp:
                            wsd[sty in self.target_concepts] += tmp[sty]
                        p = wsd[True] / (sum(wsd.values()) + 1e-5)

                        if p < self.filter_ambiguous:
                            ambiguous = True

                    # or hard filter *any* ambiguous terms
                    else:
                        for sab, sty in stys:
                            if sty not in self.target_concepts:
                                ambiguous = True
                                break

                if self.filter_ambiguous and ambiguous:
                    continue

                for sab, sty in stys:
                    if not self.target_concepts or sty in self.target_concepts:
                        specialist[abbrv].append(term)
                        break
        result = {}
        for abbrv in specialist:
            result[abbrv] = set(specialist[abbrv])
        return result


class CARDDictionary:
    def __init__(
        self, card_path: str, umls: UMLS, class_map: Mapping[str, int],
    ):
        """
        A class for loading a dictionary from the CARD tool.
        In order to use this class you must first download the CARD zip file from:
            https://sbmi.uth.edu/ccb/resources/abbreviation.htm
        """
        self.card_path = card_path
        self.umls = umls
        self.class_map = class_map

    def get_words(self) -> Dict[int, Dict[str, List[str]]]:
        vabbr: Dict[int, Dict[str, List[str]]] = collections.defaultdict(
            lambda: collections.defaultdict(list)
        )

        with zipfile.ZipFile(self.card_path, "r") as zp:
            for fname in ("VABBR_DS_beta.txt", "VABBR_CV_beta.txt"):
                with zp.open(
                    os.path.join("CARD_dataset_tools", fname), "r"
                ) as raw_f:
                    casted_raw_f = cast(
                        BinaryIO, raw_f
                    )  # TODO: Remove when mypy is upgraded
                    with io.TextIOWrapper(casted_raw_f) as f:
                        reader = csv.DictReader(f, delimiter="\t")
                        for row in reader:
                            cui = row["CUI"].upper().split("|")
                            for c in cui:
                                for sty in self.umls.get_sty_for_cui(c):
                                    label = self.class_map.get(sty)
                                    if label is None:
                                        continue
                                    vabbr[label][
                                        row["abbreviation"].upper()
                                    ].append(row["sense"])
        return {k: dict(v) for k, v in vabbr.items()}


class CTDDictionary(Dictionary[Set[str]]):
    def __init__(
        self, trove_path: str, dictionary_type: Literal["disease", "chemical"],
    ):
        self.dictionary_type = dictionary_type
        super().__init__(trove_path, "CTDDictionary" + dictionary_type.upper())

    def get_url(self) -> str:
        if self.dictionary_type == "disease":
            return "http://ctdbase.org/reports/CTD_diseases.csv.gz"
        elif self.dictionary_type == "chemical":
            return "http://ctdbase.org/reports/CTD_chemicals.csv.gz"
        else:
            raise RuntimeError(
                "Invalid dictionary type " + self.dictionary_type
            )

    def get_fieldnames(self) -> List[str]:
        if self.dictionary_type == "disease":
            return "DiseaseName,DiseaseID,AltDiseaseIDs,Definition,ParentIDs,TreeNumbers,ParentTreeNumbers,Synonyms,SlimMappings".split(  # noqa
                ","
            )
        elif self.dictionary_type == "chemical":
            return "ChemicalName,ChemicalID,CasRN,Definition,ParentIDs,TreeNumbers,ParentTreeNumbers,Synonyms".split(
                ","
            )
        else:
            raise RuntimeError(
                "Invalid dictionary type " + self.dictionary_type
            )

    def load(self) -> Set[str]:
        stopwords = get_stopwords()

        term_candidates: Set[str] = set()

        primary_field = (
            "DiseaseName"
            if self.dictionary_type == "disease"
            else "ChemicalName"
        )

        with gzip.open(self.full_path, "rt") as fp:
            filtered_file = (row for row in fp if not row.startswith("#"))
            reader = csv.DictReader(
                filtered_file, fieldnames=self.get_fieldnames()
            )

            for row in reader:
                synset = row["Synonyms"].split("|")
                term_candidates |= set(synset)

                term = row[primary_field]
                term_candidates.add(term)

        term_candidates = {
            t.strip() for t in term_candidates if t.strip() != ""
        }

        terms = {lowercase(t) for t in term_candidates}
        # filter out stopwords
        return {
            t
            for t in terms
            if t not in stopwords and not re.search(r"""^[0-9]$""", t)
        }


class BioportalDictionary(Dictionary[Collection[str]]):
    def __init__(self, trove_path: str, ontology_name: str):
        self.ontology_name = ontology_name
        super().__init__(trove_path, "CTDDictionary" + ontology_name.upper())

    def get_url(self) -> str:
        return f"http://data.bioontology.org/ontologies/{self.ontology_name}/download?apikey=8b5b7825-538d-40e0-9e9e-5ab9274a9aeb&download_format=csv"  # noqa

    def load(self) -> Collection[str]:
        stopwords = get_stopwords()
        term_candidates: Set[str] = set()

        with gzip.open(self.full_path, "rt") as fp:
            reader = csv.DictReader(fp)

            for row in reader:
                synset = row["Synonyms"].split("|")
                term_candidates |= set(synset)

                term = row["Preferred Label"]
                term_candidates.add(term)

        term_candidates = {
            t.strip() for t in term_candidates if t.strip() != ""
        }

        terms = {lowercase(t) for t in term_candidates}

        # filter out stopwords
        return {
            t
            for t in terms
            if t not in stopwords and not re.search(r"""^[0-9]$""", t)
        }
