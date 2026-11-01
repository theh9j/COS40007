# Progress

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
