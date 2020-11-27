# Few-shot Weak Supervision

## 1. Annotation Guidelines 

* Chemicals & Diseases [BC5CDR](bc5_CDR_data_guidelines.pdf)
* Disorders [ShARe/CLEF 2014](ShARe-Task-1-Guidelines-2013.pdf)
* Drugs [2009 i2b2 Medication Challenge](Preliminary.Annotation.Guidelines.6.12.pdf)

## 2. Labeling Function Examples

### Building positive/negative dictionaries

```
The patient has severe pre-eclampsia.
```
*The sentence contains only one disorder mention, “severe pre-eclempsia.” It corresponds to CUI C0341950 (preferred term: Severe pre-eclampsia). The sub-span “pre-eclampsia” can be mapped to CUI C0032914 (preferred term: Pre-eclampsia) but is not annotated as it is more general.*

