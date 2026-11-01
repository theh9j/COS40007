# Portfolio 3 - Advanced ML and Deep Learning

**Pham Quang Minh | Student ID: 104999568**

GitHub: https://github.com/theh9j/COS40007

## Current checkpoint

Iteration 1: task review, project setup and data checks. No model has been trained.
Work will be completed and recorded in small, genuine checkpoints.

## Chosen direction

- Dataset: the same Concrete Compressive Strength data used in Portfolios 1-2.
- Route: **3d - ANN for non-sequential, tabular data**.
- Working target: the four strength classes from Portfolio 2.
- Part A: Random Forest Classifier and scikit-learn MLPClassifier.
- Part B: a Keras ANN, followed by one controlled variation.
- Main classification measures: accuracy, weighted F1 and five-fold CV.

The concrete age column does not make the rows a time series. CNN and LSTM are
not needed for this dataset. The tabular route supports regression too; the current
classification choice continues Portfolio 2 and can be reconsidered before training.

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
2. Prepare inputs and establish the Random Forest baseline.
3. Add the scikit-learn MLP, five-fold evaluation, comparison chart and saved models.
4. Train the first ANN; inspect its summary, loss curves and predictions.
5. Change one ANN setting, compare fairly, and save the model and history.
6. Write a short report, final comparison and honest GenAI reflection.

Record each completed step in `PROGRESS.md`. Do not split finished work into
artificial commits or backdate activity. No future checkpoint is marked complete.

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

The brief restricts whole-code generation by GenAI. This checkpoint records review,
planning and data verification, not completed modelling code. Keep attribution
honest when using the supplied examples or receiving debugging help. The personal
learning-reflection answers must describe the student's actual experience.

Data: Yeh, I. (1998). Concrete Compressive Strength. UCI Machine Learning Repository.
https://doi.org/10.24432/C5PK67. CC BY 4.0.
