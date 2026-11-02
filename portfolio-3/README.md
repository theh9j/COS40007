# Portfolio 3 - Advanced ML and Deep Learning

**Pham Quang Minh | Student ID: 104999568**

GitHub: https://github.com/theh9j/COS40007

## Current checkpoint

Iteration 6: four model runs, final interpretation and the seven-page PDF review draft are ready.

| Model | Test accuracy | Weighted F1 | Mean CV accuracy |
| --- | ---: | ---: | ---: |
| Random Forest | 78.1% | 0.781 | 69.2% |
| MLP | 82.1% | 0.820 | 75.2% |
| ANN, up to 10 epochs | 62.3% | 0.583 | Not run |
| ANN, up to 50 epochs | 75.7% | 0.746 | Not run |

MLP is the current traditional baseline selected by CV accuracy. All six MLP fits
(five CV fits and the final fit) reached 500 iterations without convergence.
The warnings are recorded; these fixed baseline settings were not tuned using
test scores. The 50-epoch ANN has 72.4% validation accuracy and test loss 0.602.
It is preferred over the 10-epoch ANN by validation loss, but the MLP retains the
highest observed test accuracy. The report is drafted and checked; the student-supplied reflection is included and student permitted-use review remains pending.

## Chosen direction

- Dataset: the same Concrete Compressive Strength data used in Portfolios 1-2.
- Route: **3d - ANN for non-sequential, tabular data**.
- Confirmed target: the four strength classes from Portfolio 2.
- Part A: Random Forest Classifier and scikit-learn MLPClassifier.
- Part B: a Keras ANN, followed by one controlled variation.
- Main classification measures: accuracy, weighted F1 and five-fold CV.

The concrete age column does not make the rows a time series. CNN and LSTM are
not needed for this dataset. Classification continues the agreed Portfolio 2 task.

## Verified data

`data/concrete_partitioned.csv` is an unchanged copy of Portfolio 2's prepared data.
It has 1,005 records, eight original inputs, continuous strength, four class labels,
source row IDs, recipe IDs and the saved train/test partition.

- Training: 676 records; 169 in each class.
- Testing: 329 records.
- Missing values: none.
- Recipes shared between training and testing: none.

Use only the eight ingredient/age inputs for the initial comparison. Strength,
class labels, source IDs, recipe IDs and partition labels must not enter the
predictor matrix. Keep the same outer split for Parts A and B. Fit StandardScaler
only on training data and within CV folds. The ANN fits on 513 rows, with 163 rows
reserved from training for validation (20% of training recipe groups). Its scaler
uses only those 513 fitting rows. Earlier portfolios already inspected this
dataset, so the test set is a comparison set, not new external validation.

## Small iterations

1. **Done:** agree the route, verify the data and create the notebook outline.
2. **Done:** prepare inputs, evaluate the Random Forest and save its pipeline.
3. **Done:** add the MLP using the same five folds and create the comparison chart.
4. **Done:** train the first ANN; inspect its summary, curves and predictions.
5. **Done:** keep its split, seed and settings fixed; increase maximum epochs to 50.
6. **Done:** short PDF, final comparison, transparent GenAI declaration and student-supplied personal reflection. Review permitted use before submission.

Record each completed step in `PROGRESS.md`. The user writes commit messages and
handles commits/pushes; the assistant must not stage, commit or push. No future
checkpoint is marked complete.

## Run and inspect

Install `requirements.txt` in a Python 3.12 environment, then open the notebook with
your Jupyter editor, restart the kernel and run all cells in order from `portfolio-3`. The dataset is
local; no dataset download is required. TensorFlow 2.21.0 and Keras 3.15.1 are
verified on CPU. The locally installed `.ann-runtime/` is excluded from Git;
other computers should install the pinned requirements in their own environment.

- `results/model_comparison.csv`: both traditional models and both ANN runs.
- `results/random_forest_cv.csv`: the five individual CV scores.
- `results/random_forest_test_predictions.csv`: predictions for the test records.
- `figures/random_forest_confusion_matrix.png`: errors by strength class.
- `models/random_forest_baseline.pkl`: fitted scaler and classifier together.
- `checks/cv_recipe_folds.csv`: fixed fold assignments shared by both models.
- `checks/iteration-02-checks.json`: parameters, versions and verification results.
- `results/mlp_cv.csv`: MLP fold scores, iteration counts and final training loss.
- `results/mlp_test_predictions.csv`: MLP predictions for the test records.
- `results/mlp_training_loss.csv`: the final MLP training loss by iteration.
- `models/mlp_baseline.pkl`: fitted MLP and its scaler together.
- `figures/traditional_models_comparison.png`: comparison chart and MLP confusion matrix.
- `checks/iteration-03-checks.json`: MLP settings, metrics, warnings and checks.
- `checks/ann_experiment_plan.json`: baseline and next variation fixed before ANN test evaluation.
- `checks/ann_recipe_split.csv`: ANN fitting/validation source rows and recipe groups.
- `checks/iteration-04-checks.json`: ANN settings, metrics, versions and checks.
- `models/ann_baseline_10epochs.h5`: validation-selected ANN in the requested format.
- `models/ann_baseline_best.keras`: native Keras checkpoint of the same selected model.
- `models/ann_baseline_scaler.pkl`: scaler fitted on ANN fitting rows only.
- `results/ann_baseline_history.csv` and `.pkl`: all ten epochs of training history.
- `results/ann_baseline_test_predictions.csv`: class predictions and probabilities.
- `results/ann_baseline_classification_report.csv`: per-class results.
- `figures/ann_baseline_learning_curves.png`: training and validation loss/accuracy.
- `figures/ann_baseline_comparison.png`: three-model test comparison and ANN confusion matrix.
- `checks/iteration-05-checks.json`: controlled-change checks and extended ANN results.
- `results/ann_variants_comparison.csv`: the two ANN runs side by side.
- `models/ann_extended_50epochs.h5` and `ann_extended_best.keras`: selected extended ANN.
- `results/ann_extended_history.csv` and `.pkl`: all 50 epochs of history.
- `results/ann_extended_test_predictions.csv`: extended ANN predictions and probabilities.
- `results/ann_extended_classification_report.csv`: per-class extended ANN results.
- `figures/ann_extended_learning_curves.png`: extended training and validation curves.
- `figures/four_model_comparison.png`: all four test scores and extended ANN confusion matrix.

The baseline uses 200 trees and at least two records per leaf. Settings were fixed
before evaluating test predictions; no hyperparameter tuning was done. The scaler
is fitted separately within each CV fold and on the training partition for the
final fit. Training accuracy is 97.2%, so its training fit is optimistic.

The MLP uses the teaching example's 100/50 hidden layers, ReLU, Adam and seed 42.
Scaling is fitted inside each training fold. The final MLP training accuracy is
97.3%, which is also optimistic compared with CV performance.

The ANN has 64/32/16 ReLU hidden units, dropout 0.2 after the first two layers and
four softmax outputs (3,252 trainable parameters). Adam starts at 0.001; batch size
is 32. Callbacks select the lowest validation loss, stop after 10 unimproved epochs
and halve the learning rate after 5. All ten epochs completed. Validation loss fell
through epoch 10; this is an early training result rather than a converged-model claim.
ANN and traditional test scores are descriptive comparisons: the ANN holds out
validation rows, whereas the traditional models fit all 676 training rows.

The second ANN starts from the same seed and changes only the maximum epochs to
50. All 50 epochs completed; epoch 50 had the lowest validation loss. The first ten
epochs reproduce the baseline history. Validation loss improved from 0.908 to
0.652 and test accuracy rose by 13.4 percentage points. No extra settings were tried
after inspecting these test results. Earlier model files and results are preserved.

## Deliverables and final review

- One notebook: `Portfolio_Assessment_3_104999568_Pham_Quang_Minh.ipynb`.
- One concise PDF with the Portfolio 1 cover style, student ID and GitHub link.
- Saved traditional models (`.pkl`), ANN model (`.h5`) and training history.
- Labelled comparison charts, ANN curves and confusion matrix.

## References and AI use

The attached brief, rubric and teaching notebooks are references. The supplied
classification MLP example uses Iris, while the ANN examples use synthetic data;
they need appropriate adaptation before use with concrete data. In particular,
the ANN classification example has a binary output, whereas this task has four
classes. The tabular brief specifies Dense layers of 64, 32 and 16 units with
dropout, followed by a four-class softmax output.

The brief restricts whole-code generation by GenAI. Checkpoints 2-5 include
AI-assisted baseline implementation, MLP/ANN example adaptation and verification, recorded in
the notebook and progress log. Review the implementation and its permitted use
before submission. The student-supplied reflection is included, with grammar edits only.

Data: Yeh, I. (1998). Concrete Compressive Strength. UCI Machine Learning Repository.
https://doi.org/10.24432/C5PK67. CC BY 4.0.

## Report draft and checks (2 November 2026)

The seven-page `Portfolio_Assessment_3_104999568_Pham_Quang_Minh.pdf` follows
Portfolio 1's cover, fonts, blue headings and red footer. Its build source is
`tools/build_report.py` (ReportLab); it reads saved results without retraining.
`checks/iteration-06-checks.json` records notebook validation, preserved code/output
cells, unchanged dataset hash and independently recomputed test metrics.

The report now includes your personal reflection, lightly edited for grammar. Review
the declared AI implementation assistance against course rules before submission.
No staging, commit or push was performed. Git remains under your control.
