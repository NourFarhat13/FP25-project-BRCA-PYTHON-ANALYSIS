# Breast Cancer Survival Analysis

Fundamentals of Programming (FP25) —  project.

## Overview

Exploratory analysis of a breast cancer dataset (TCGA) examining whether
clinical and biomarker factors are associated with patient survival.
Includes descriptive statistics, visualizations, and K-Means clustering.

## Project Structure

| File               | Description                                      |
|--------------------|--------------------------------------------------|
| `raw_data/BRCA.csv`| Original dataset (334 patients)                  |
| `process_data.py`  | Loads and cleans raw data → `data_full.csv`      |
| `data_full.csv`    | Cleaned dataset (321 patients)                   |
| `functions.py`     | Reusable functions for analysis and visualization|
| `test_functions.py`| Unit tests for `functions.py`                    |
| `main_file.qmd`    | Quarto report (source)                           |
| `requirements.txt` | Python dependencies                              |

## How to Reproduce

```bash
pip install -r requirements.txt
python process_data.py
quarto render main_file.qmd
```

## Requirements

- Python 3.10+
- Quarto CLI

## Note:
 If quarto render has issues finding Python packages,
adding engine: knitr to the YAML header resolves this
(requires R with the reticulate package) 