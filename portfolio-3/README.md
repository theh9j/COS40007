# Portfolio 3 - Advanced ML and Deep Learning

**Pham Quang Minh | Student ID: 104999568**

GitHub: https://github.com/theh9j/COS40007

## Current checkpoint

Iteration 2: inputs prepared and the Random Forest baseline trained and evaluated.
Test accuracy is **78.1%**, weighted F1 is **0.781**, and mean five-fold recipe-group
CV accuracy is **69.2%**. These are baseline results, not a final model selection.

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
only on training data and within CV folds. Later, reserve recipe groups from the
training partition for ANN validation. Earlier portfolios already inspected this
dataset, so the test set is a comparison set, not new external validation.

## Small iterations

1. **Done:** agree the route, verify the data and create the notebook outline.
2. **Done:** prepare inputs, evaluate the Random Forest and save its pipeline.
3. Add the scikit-learn MLP using the same five folds, then create the comparison chart.
4. Train the first ANN; inspect its summary, loss curves and predictions.
5. Change one ANN setting, compare fairly, and save the model and history.
6. Write a short report, final comparison and honest GenAI reflection.

Record each completed step in `PROGRESS.md`. The user writes commit messages and
handles commits/pushes; the assistant must not stage, commit or push. No future
checkpoint is marked complete.

## Run and inspect

Install `requirements.txt` in a Python 3.12 environment, then open the notebook with
your Jupyter editor and run its cells in order from `portfolio-3`. The dataset is
local; no download is required. TensorFlow is not needed for this checkpoint.

- `results/model_comparison.csv`: the first model's accuracy and weighted F1 results.
- `results/random_forest_cv.csv`: the five individual CV scores.
- `results/random_forest_test_predictions.csv`: predictions for the test records.
- `figures/random_forest_confusion_matrix.png`: errors by strength class.
- `models/random_forest_baseline.pkl`: fitted scaler and classifier together.
- `checks/cv_recipe_folds.csv`: fixed fold assignments for the upcoming MLP.
- `checks/iteration-02-checks.json`: parameters, versions and verification results.

The baseline uses 200 trees and at least two records per leaf. Settings were fixed
before evaluating test predictions; no hyperparameter tuning was done. The scaler
is fitted separately within each CV fold and on the training partition for the
final fit. Training accuracy is 97.2%, so its training fit is optimistic.

## Deliverables to build

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

The brief restricts whole-code generation by GenAI. Checkpoint 2 includes
AI-assisted baseline implementation, explanations and verification, recorded in
the notebook and progress log. Review the implementation and its permitted use
before submission. Personal learning reflections remain for the student to write.

Data: Yeh, I. (1998). Concrete Compressive Strength. UCI Machine Learning Repository.
https://doi.org/10.24432/C5PK67. CC BY 4.0.
