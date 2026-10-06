# Portfolio 1 - James Jackson

Student ID: **104999568**. COS40007 Artificial Intelligence Engineering.

## Main submission

- `Portfolio_Assessment_1_James_Jackson.ipynb`: single executed notebook, Tasks 1-9.
- `Portfolio_Assessment_1__JamesJackson.pdf`: report summary and
  the complete notebook, with all code, recorded outputs, figures and reasoning.

The studio field is intentionally blank. Supply it and rename the PDF to the required
`Portfolio_Assessment_1_[Studio_Number]_JamesJackson.pdf` before submission.

## Review before submission

The GenAI declaration honestly describes substantial AI drafting. The supplied brief
permits narrower assistance. Verify all work, confirm its permitted use with the
course, and do not claim AI-authored interpretations as unaided original work.
The notebook records descriptive findings, not structural design recommendations.

## Run

Use Python 3.12 or a compatible version. From this directory:

```shell
python -m venv .venv
# Activate .venv using the command appropriate for your operating system.
python -m pip install -r requirements.txt
python -m pip install jupyterlab
python -m jupyter lab
```

Open the notebook and run all cells in order. Data is included for offline use.
The build script is an auxiliary exporter, not a second assignment submission:
`python build_portfolio.py` executes the notebook and rebuilds the PDF. It uses a
temporary local kernelspec and prefers `.runtime` packages if present.

## Data provenance

Yeh, I. (1998). Concrete Compressive Strength [Dataset]. UCI Machine Learning
Repository. DOI: https://doi.org/10.24432/C5PK67. CC BY 4.0.
Downloaded 6 October 2026 from the UCI archive. The original workbook and README
are under `data/`. `concrete_raw.csv` preserves the original rows with short column
names; `concrete_prepared.csv` removes exact full-record repetitions and adds four
relative quartile classes. Metadata records units, source hash and class boundaries.

## GitHub Classroom

Invitation from the brief: https://classroom.github.com/a/775q4tU1.
Accept the invitation using your own account, then provide the assigned repository
URL to upload the work. No GitHub push has been performed. Do not upload `.runtime`,
`.cache` or `.kernels`; `.gitignore` excludes them. Submit the notebook with its data
and requirements so another person can reproduce the analysis.
