# Progress

## Iteration 6 - 2 November 2026

- Created a seven-page PDF review draft in Portfolio 1 style, with simple explanations.
- Added the final comparison, ANN curves, architecture, confusion matrix and references.
- Declared implementation/reporting assistance honestly; no personal experience invented.
- Updated notebook conclusions and removed stale future-checkpoint wording.
- Preserved all 20 executed code cells and their outputs; no retraining.
- Validated notebook structure, dataset hash, all four test accuracy/F1 scores and all rendered PDF pages.
- Added a reproducible PDF builder and final check record.

**Reflection update (2 November):** added the student-supplied account of the ANN
setup difficulty, lightly edited for grammar, to the notebook and PDF. No extra
personal experience was invented. Student permitted-use review remains.

---

## Iteration 5 - 1 November 2026

**Completed**

- Ran the previously specified 50-epoch ANN variation from the same initial seed.
- Kept the 513/163/329 split, scaler, layers, dropout, batch size and callbacks fixed.
- Verified that all recorded metrics in the first ten epochs match the original run.
- Selected epoch 50 and the extended ANN using validation loss, before test evaluation.
- Saved the extended H5/Keras model, history, predictions, classification report and plots.
- Updated the master results table and added the two-ANN comparison table.
- Preserved earlier notebook outputs and model/result files; only new training was run.

| Measure | 10-epoch ANN | 50-epoch ANN |
| --- | ---: | ---: |
| Test accuracy | 62.31% | 75.68% |
| Test weighted F1 | 0.5830 | 0.7462 |
| Validation accuracy | 60.12% | 72.39% |
| Validation loss | 0.9078 | 0.6525 |
| Test loss | 0.9325 | 0.6023 |
| Selected epoch | 10 | 50 |

All 50 epochs completed; early stopping did not end the run early. More training
improved this ANN, but the MLP still has the highest observed test accuracy (82.07%).
This is not a controlled architecture comparison: traditional models use more
fitting records, and the MLP's convergence warnings remain recorded.

**Checks:** matching first-ten-epoch history; original split, scaler and all prior
model/prediction/history hashes unchanged; H5 reload reproduces predictions;
history reload matches; saved probabilities and metrics checked independently.
The five new notebook code cells executed successfully. Earlier cell outputs were
preserved; restarting the kernel and running all cells reproduces the full workflow.

**Next checkpoint:** concise PDF in the Portfolio 1 layout, final file checks and
the student's own personal learning reflection. No more model runs are planned.

---

## Iteration 4 - 1 November 2026

**Completed**

- Installed and verified TensorFlow 2.21.0 / Keras 3.15.1 in an ignored, separate runtime.
- Kept the original 329 test records; split training recipes into 513 fitting rows
  (239 recipes) and 163 validation rows (60 recipes), using seed 43.
- Fitted a separate StandardScaler only on the 513 ANN fitting rows.
- Built the specified 64/32/16 ReLU ANN, two dropout layers (0.2) and four-class softmax.
- Trained 10 epochs with batch size 32, Adam 0.001 and validation-loss callbacks.
- Selected epoch 10 by the lowest validation loss; no test-based model selection.
- Saved H5/native Keras models, scaler, history, predictions, report and two figures.
- Fixed the next experiment in advance: maximum 50 epochs, otherwise identical settings.

| Measure | Result |
| --- | ---: |
| ANN test accuracy | 62.31% |
| ANN test weighted F1 | 0.5830 |
| ANN validation accuracy | 60.12% |
| ANN test loss | 0.9325 |
| Selected epoch / completed epochs | 10 / 10 |

Training and validation losses improved overall; the short ANN run remains weaker
than the MLP baseline. Validation is a separate split, not ANN five-fold CV. The
ANN uses fewer fitting records because it reserves validation rows.

**Checks:** original data/fold/prediction hashes and traditional baseline metrics
remain unchanged. ANN fit/validation/test recipes are disjoint; scaler statistics
use fitting rows only; H5 reload reproduces predictions; saved history matches;
probabilities are finite and sum to one. All 15 code cells completed without an
execution error. The displayed training curve uses dropout, while validation and
the saved model's evaluation do not.

**Next checkpoint:** run the pre-specified 50-epoch ANN variation, compare validation
behaviour and saved model results. Final report and personal reflection remain pending.

---

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
