# Portfolio 1 - Pham Quang Minh

Student ID: **104999568**. COS40007 Artificial Intelligence Engineering.

## Main submission

- `Portfolio_Assessment_1_Pham_Quang_Minh.ipynb`: single executed notebook, Tasks 1-9.
- `Portfolio_Assessment_1__Pham_Quang_Minh.pdf`: report summary and
  the complete notebook, with all code, recorded outputs, figures and reasoning
  
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
