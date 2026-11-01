# Portfolio 3 - Advanced ML and Deep Learning

**Pham Quang Minh | Student ID: 104999568**

GitHub: https://github.com/theh9j/COS40007

## Current checkpoint

Iteration 3: Random Forest and MLP baselines are trained and compared.

| Model | Test accuracy | Weighted F1 | Mean CV accuracy |
| --- | ---: | ---: | ---: |
| Random Forest | 78.1% | 0.781 | 69.2% |
| MLP | 82.1% | 0.820 | 75.2% |

MLP is the current traditional baseline selected by CV accuracy. All six MLP fits
(five CV fits and the final fit) reached 500 iterations without convergence.
The warnings are recorded; these fixed baseline settings were not tuned using
test scores. Keras ANN experiments are not complete yet.

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
3. **Done:** add the MLP using the same five folds and create the comparison chart.
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

- `results/model_comparison.csv`: both models' accuracy and weighted F1 results.
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

The baseline uses 200 trees and at least two records per leaf. Settings were fixed
before evaluating test predictions; no hyperparameter tuning was done. The scaler
is fitted separately within each CV fold and on the training partition for the
final fit. Training accuracy is 97.2%, so its training fit is optimistic.

The MLP uses the teaching example's 100/50 hidden layers, ReLU, Adam and seed 42.
Scaling is fitted inside each training fold. The final MLP training accuracy is
97.3%, which is also optimistic compared with CV performance.

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

The brief restricts whole-code generation by GenAI. Checkpoints 2-3 include
AI-assisted baseline implementation, MLP example adaptation and verification, recorded in
the notebook and progress log. Review the implementation and its permitted use
before submission. Personal learning reflections remain for the student to write.

Data: Yeh, I. (1998). Concrete Compressive Strength. UCI Machine Learning Repository.
https://doi.org/10.24432/C5PK67. CC BY 4.0.
