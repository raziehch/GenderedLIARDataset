# GenderedLIAR Dataset

**GenderedLIAR** is an augmented version of the [LIAR dataset (Wang, 2017)](https://aclanthology.org/P17-2067/) for studying gender bias in fake news detection.

Each statement is paired with **gender-variant rewrites of the speaker's job title** — neutral, male, and female. Only the job title changes while the statement stays fixed, giving a controlled test of whether a classifier's prediction shifts with the speaker's perceived gender.

- **Paper**: [Unequal Verdicts: Investigating Gender Bias in LLM-Based Fake News Detection](https://arxiv.org/abs/2608.03627)
- **File**: `GenderedLIARDataset.csv`
- **Rows**: 9,223 — the LIAR statements that have a speaker job title recorded
- **Columns**: 25
- **Base dataset**: [LIAR](https://aclanthology.org/P17-2067/) (Wang, 2017)

## Columns

Variants are formed either by inserting a gender word before the role noun (`Bexar County Judge` → `Bexar County Male Judge`) or, where English supplies gendered forms, by lexical substitution (`Chair` → `Chairman`/`Chairwoman`, `Spokesperson` → `Spokesman`/`Spokeswoman`). All three variant columns are fully populated — where the original title is already neutral, male, or female, that variant equals `speaker_job_cased`. The exception is `Plural` and `NaP` rows, which have no variants at all (491 rows, all three columns blank) and are removed by the filters below.

| Column | Description |
|---|---|
| `statement` | The statement being evaluated. |
| `speaker_job_cased` | Original LIAR job title, title-cased. |
| `speaker_job_neutral_cased` | Gender-neutral variant of the job title. `NotFound` where no neutral form could be derived (these rows are removed by the filters). |
| `speaker_job_male_cased` | Male variant of the job title. |
| `speaker_job_female_cased` | Female variant of the job title. |
| `speaker_gender_label` | Gendered nature of the original title: `Male`, `Female`, `Neutral`, `Plural` (refers to more than one person), `NaP` (not a person). |
| `statement_reveals_male` / `_female` | `Yes`/`No` — whether the statement text itself discloses the speaker's gender. |
| `speaker_job_reveals_name_direct` / `_indirect` | `Yes`/`No` — whether the job title reveals the speaker's name, which could reveal gender. |
| `label_binary` | Binarized label: `True` ← true, mostly-true, half-true; `False` ← barely-true, false, pants-fire. |

All other LIAR columns are preserved unchanged: `ID`, `label`, `subjects`, `speaker`, `speaker_job`, `state`, `party`, `context`, `split`, and the speaker credit-history counts.

## Reproducing the paper's results

All results in the paper use the **filtered** set. The filters drop rows whose gender is disclosed by other means, which would confound the job-title manipulation.

```python
import pandas as pd

df = pd.read_csv("GenderedLIARDataset.csv")

filtered_df = df[
    (df['statement_reveals_male'] == 'No') &
    (df['statement_reveals_female'] == 'No') &
    (df['speaker_job_reveals_name_direct'] == 'No') &
    (df['speaker_job_reveals_name_indirect'] == 'No') &
    (df['speaker_job_neutral_cased'] != 'NotFound') &
    (df['speaker_gender_label'].isin(['Male', 'Female', 'Neutral']))
]
```

## Statistics

Full dataset, all five `speaker_gender_label` categories (n = 9,223):

| `speaker_gender_label` | Count | Share |
|---|---|---|
| Neutral | 8,111 | 87.9% |
| Male | 507 | 5.5% |
| Plural | 311 | 3.4% |
| NaP | 180 | 2.0% |
| Female | 114 | 1.2% |
| **Total** | **9,223** | **100%** |

Filtered set (n = 8,243):

| `speaker_gender_label` | Count | Share | True | False |
|---|---|---|---|---|
| Neutral | 7,681 | 93.2% | 4,504 (58.6%) | 3,177 (41.4%) |
| Male | 452 | 5.5% | 254 (56.2%) | 198 (43.8%) |
| Female | 110 | 1.3% | 47 (42.7%) | 63 (57.3%) |
| **Total** | **8,243** | **100%** | **4,805 (58.3%)** | **3,438 (41.7%)** |

## Citation

Accepted for presentation at the 6th Workshop on Bias and Fairness in AI at ECML PKDD 2026 (Naples, Italy, 7 September 2026). Please cite the arXiv preprint until the proceedings version is available:

```bibtex
@misc{chalehchaleh2026unequalverdictsinvestigatinggender,
  title         = {Unequal Verdicts: Investigating Gender Bias in LLM-Based Fake News Detection},
  author        = {Razieh Chalehchaleh and Reza Farahbakhsh and Noel Crespi},
  year          = {2026},
  eprint        = {2608.03627},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/abs/2608.03627}
}
```
