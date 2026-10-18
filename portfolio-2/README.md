# Portfolio 2 - Pham Quang Minh

COS40007 Artificial Intelligence Engineering, 18 October 2026.
Student ID: **104999568**. The studio number is left blank. The country acknowledgement is omitted
as requested. Portfolio 1 is preserved unchanged.

## Deliverables

- `Portfolio_Assessment_2_Pham_Quang_Minh.ipynb`: self-contained executed notebook.
- `Portfolio_Assessment_2__Pham_Quang_Minh.pdf`: Portfolio 1-style cover and report, comparison tables
  and complete executed notebook with all code, recorded outputs and figures.
- `portfolio-2-submission.zip`: portable submission files; excludes build tools,
  temporary review images, caches, kernels and environment packages.

The empty field between underscores in the PDF filename is the studio number.
Rename it when the number is available. No GitHub upload has been performed.
The assignment's Classroom invitation is https://classroom.github.com/a/ifqEa70j .

## Run the notebook

Use Python 3.12 and install `requirements.txt` in a virtual environment. Install
JupyterLab separately if needed. Open the notebook from this folder and run all
cells in order. The original UCI workbook is included; internet is not required.
Nested group cross-validation performs many small fits, so allow several minutes.

```shell
python -m venv .venv
# Activate your environment, then:
python -m pip install -r requirements.txt
python -m pip install jupyterlab
python -m jupyter lab
```

`tools/build_portfolio.py` is an optional local regeneration helper, not a second
assignment submission. It can reuse the existing Portfolio 1 package runtime in
this workspace, or use normally installed packages elsewhere. Run it from Python
to recreate the notebook, execute it and export the PDF. `--export-only` exports
the current executed notebook without rerunning models. The notebook contains all
analysis code and does not import the helper.

## Method

The source has 1,030 rows; 25 extra exact duplicates are removed. Recipe groups
are defined by the seven ingredient quantities, excluding age. A seeded 70/30
split of groups reserves unseen recipes. Because groups have different sizes,
the row percentages may differ from 70/30. All model comparisons use the same
partition and five group folds on the training data.

Four relative strength classes use training quartiles. A custom estimator learns
the boundaries separately within every inner and outer fit; validation targets
are labelled using the fitted edges. Transformations and feature selection also
stay within training. These are not structural strength grades. Fold-specific
boundaries may differ slightly. The holdout labels use full-training boundaries
shared by all fitted models. The previous full-data EDA informed feature hypotheses,
so external validation remains necessary.

Five fixed decision-tree feature sets are compared. Four final SVM variations are
baseline, tuned, K-Best and PCA. Tuned variants use five inner group folds and
five outer group folds. CV macro F1 selects models; the holdout is not used to
choose features or parameters. A separate unscaled SVM checks scaling's effect.

`results/` contains both comparison tables, all 45 main outer-fold scores,
the tuning grid, holdout predictions and reproducibility metadata. `data/` includes
the original workbook/README and a partition audit with source row indices.
`figures/` contains standalone labelled charts.

## Attribution and review

Yeh, I. (1998). Concrete Compressive Strength [Dataset]. UCI Machine Learning
Repository. https://doi.org/10.24432/C5PK67 . CC BY 4.0. The source was copied from
Portfolio 1, which recorded a UCI download on 6 October 2026.

An honest GenAI declaration appears at the beginning of the notebook and report.
Review and verify the work before submission. The supplied course brief describes
narrower permitted uses, so confirm that the declared level of assistance is
acceptable. The report does not claim to complete the in-person studio demo.
