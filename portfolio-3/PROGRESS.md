# Progress

## Iteration 3 - 1 November 2026

**Completed**

- Adapted the supplied Iris MLP example to the concrete classification task.
- Added a scaled MLP with 100/50 hidden units, ReLU, Adam and up to 500 iterations.
- Reused the same inputs, labels, test partition and five saved recipe folds.
- Compared both models and saved the MLP pipeline, predictions and training loss.
- Added a two-model accuracy chart, MLP confusion matrix and verification record.
- Selected MLP as the traditional baseline by mean CV accuracy, before ANN comparison.

| Model | Test accuracy | Test weighted F1 | Mean CV accuracy |
| --- | ---: | ---: | ---: |
| Random Forest | 78.12% | 0.7805 | 69.16% |
| MLP | 82.07% | 0.8203 | 75.18% |

**Limitation:** all five CV MLP fits and the final fit emitted convergence warnings
at the fixed 500-iteration limit. The model produced valid predictions, but full
optimisation is not established. Warnings and iteration counts are saved, not
silently discarded. No settings were changed to improve the test score.

**Checks:** training-only scaler means checked for every fit; saved model predictions
match; source data and fold assignments unchanged; both comparison rows checked
against predictions and CV results. All nine notebook code cells completed without
execution errors. Random Forest metrics remain unchanged from iteration 2.

**Next checkpoint:** prepare recipe-group ANN validation data and train the first
Keras ANN. TensorFlow setup and the final report remain pending.

**AI assistance:** adapting the supplied example, explanations, plots and verification.
The personal learning reflection remains for the student to write.

**Git:** nothing staged, committed or pushed; changes left for the user's review.

---

## Iteration 2 - 1 November 2026

**Completed**

- Added and executed five notebook code cells for data preparation and the Random Forest.
- Kept eight original inputs, existing class labels and the 676/329 recipe split.
- Added StandardScaler inside the model pipeline; each CV fold learns its own scaler.
- Trained a fixed baseline: 200 trees, minimum two records per leaf, seed 42.
- Ran five-fold recipe-group CV and evaluated the separate test set.
- Saved the model pipeline, results, test predictions, confusion matrix and CV assignments.
- Added pinned requirements and a verification record for this checkpoint.

**Results**

| Measure | Result |
| --- | ---: |
| Test accuracy | 78.12% |
| Test weighted F1 | 0.7805 |
| Mean CV accuracy | 69.16% |
| Mean CV weighted F1 | 0.6925 |
| Training accuracy | 97.19% |

Training performance is optimistic. This is one baseline, so no best-model claim is made.

**Checks passed:** original data hash unchanged; no shared train/test or CV recipes;
all scalers fitted only on their training rows; reloaded model predictions exactly
match the saved model's predictions; each training record appears in one CV
validation fold. The notebook completed without execution errors.

**Next checkpoint:** add the scikit-learn MLP and compare both models using the same
saved folds. ANN training and the final PDF remain pending.

**AI assistance:** baseline code drafting, explanations and verification. No personal
learning reflection was invented. The supplied brief's limits on AI use still apply.

**Git:** changes left unstaged and uncommitted for the user. No push performed.

---

## Iteration 1 - 1 November 2026

**Completed**

- Reviewed the task, rubric, four route documents and five example notebooks.
- Chose the tabular ANN route (3d), provisionally continuing the four-class task.
- Set identity to Pham Quang Minh, 104999568 and the existing GitHub repository.
- Copied the prepared dataset without changing its rows, labels or split.
- Checked schema, missing values, duplicate source IDs, class counts and recipe separation.
- Created the main notebook outline and a small-step work plan.

**Verified outcome:** 1,005 rows; 676 training / 329 test; no missing values;
no shared train/test recipes. Each training class has 169 records.
The exact checks and source hashes are in `checks/iteration-01-data-audit.json`.

**Not done yet:** input scaling, model training, model evaluation and final PDF.
TensorFlow/Keras is not installed in the inspected local runtime; choose and verify
the ANN environment when that step is reached. No package installation was attempted.

**Next checkpoint:** prepare the eight original inputs and establish the Random
Forest baseline, with transformations fitted on training data only.

**AI assistance this iteration:** reviewing supplied material, planning the work,
drafting project notes and checking the existing dataset. No model results or
personal learning reflections were invented.
