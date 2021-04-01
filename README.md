# Trove 
<!--[![Build Status](https://travis-ci.com/som-shahlab/trove.svg?branch=main)](https://travis-ci.com/som-shahlab/trove)-->
[![Documentation Status](https://readthedocs.org/projects/trove/badge/?version=latest)](https://trove.readthedocs.io/en/latest/?badge=latest)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Trove is a research framework for building weakly supervised (bio)medical named entity recognition (NER) and other entity attribute classifiers without hand-labeled training data.

The COVID-19 pandemic has underlined the need for faster, more flexible ways of building and sharing state-of-the-art NLP/NLU tools to analyze electronic health records, scientific literature, and social media. Likewise, recent research into language modeling and the dangers of uncurated, ["unfathomably"](https://faculty.washington.edu/ebender/papers/Stochastic_Parrots.pdf) large-scale training data underlines the broader need to approach training set creation iself with more transparency and rigour.  

Trove provides tools for combining freely available supervision sources such as medical ontologies from the [Unified Medical Language System (UMLS)](https://www.nlm.nih.gov/research/umls/index.html), common text heuristics, and other noisy labeling sources for use as entity *labelers* in weak supervision frameworks such as [Snorkel](https://github.com/snorkel-team/snorkel), [FlyingSquid ](https://github.com/HazyResearch/flyingsquid) and others. Technical details are available in our [manuscript](https://www.nature.com/articles/s41467-021-22328-4).



Trove has been used as part of several COVID-19 reseach efforts at Stanford. 

- [Continuous symptom profiling of patients screened for SARS-CoV-2](https://med.stanford.edu/covid19/research.html#data-science-and-modeling). We used a daily feed of patient notes from Stanford Health Care emergency departments to generate up-to-date [COVID-19 symptom frequency](https://docs.google.com/spreadsheets/d/1iZZvbv94fpZdC6XaiPosiniMOh18etSPliAXVlLLr1w/edit#gid=344371264) data. Funded by the [Bill & Melinda Gates Foundation](https://www.gatesfoundation.org/about/committed-grants/2020/04/inv017214).
- [Estimating the efficacy of symptom-based screening for COVID-19](https://rdcu.be/chSrv) published in *npj Digitial Medicine*.
- Our COVID-19 symptom data was used by CMU's [DELPHI group](https://covidcast.cmu.edu/) to prioritize selection of informative features from [Google's Symptom Search Trends dataset](https://github.com/GoogleCloudPlatform/covid-19-open-data/blob/main/docs/table-search-trends.md).


## Getting Started

### Tutorials

See [`tutorials/`](https://github.com/som-shahlab/trove/tree/dev/tutorials) for Jupyter notebooks walking through an example NER application.

### Installation

Requirements: Python 3.6 or later. We recomend using `pip` to install 

`pip install -r requirements.txt`

## Contributions
We welcome all contributions to the code base! Please submit a pull request and/or start a discussion on GitHub Issues.

Weakly supervised methods for programatically building and maintaining training sets provides new opportunities for the larger community to participate in the creation of important datasets. This is especially exciting in domains such as medicine, where sharing labeled data is often challening due to patient privacy concerns.

Inspired by recent efforts such as [HuggingFace's Datasets](ttps://github.com/huggingface/datasets) library,
we would love to start a conversation around how to support sharing labelers in service of mantaining an open task library, so that it is easier to create, deploy, and version control weakly supervised models. 


## Citation
If use Trove in your research, please cite us!

Fries, J.A., Steinberg, E., Khattar, S. et al. Ontology-driven weak supervision for clinical entity classification in electronic health records. Nat Commun 12, 2017 (2021). https://doi-org.stanford.idm.oclc.org/10.1038/s41467-021-22328-4

```
@article{fries2021trove,
  title={Ontology-driven weak supervision for clinical entity classification in electronic health records},
  author={Fries, Jason A and Steinberg, Ethan and Khattar, Saelig and Fleming, Scott L and Posada, Jose and Callahan, Alison and Shah, Nigam H},
  journal={Nature Communications},
  volume={12},
  number={1},
  year={2021},
  publisher={Nature Publishing Group}
}
```


