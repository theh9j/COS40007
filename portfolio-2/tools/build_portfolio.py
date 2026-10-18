"""Generate and execute the self-contained Portfolio 2 notebook, then export PDF."""
from pathlib import Path
import sys, os, json, textwrap, re, io, base64, zipfile

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
RUNTIME = WORKSPACE / 'portfolio-1' / '.runtime'
if RUNTIME.exists():
    sys.path.insert(0, str(RUNTIME))
    os.environ['PYTHONPATH'] = str(RUNTIME)
for variable, folder in [('MPLCONFIGDIR', 'matplotlib'), ('IPYTHONDIR', 'ipython'),
                         ('JUPYTER_RUNTIME_DIR', 'jupyter')]:
    path = ROOT / '.cache' / folder
    path.mkdir(parents=True, exist_ok=True)
    os.environ[variable] = str(path)
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

cells = []
def md(text):
    cells.append(nbformat.v4.new_markdown_cell(textwrap.dedent(text).strip()))
def code(text):
    cells.append(nbformat.v4.new_code_cell(textwrap.dedent(text).strip()))

md('''# Advanced Data Preprocessing, Feature Engineering & Model Development
**Pham Quang Minh**

COS40007 Artificial Intelligence Engineering | Portfolio Assessment 2

Date: 18 October 2026 | Student ID: 104999568 | Studio number: __________

## Generative AI declaration
I used OpenAI Codex to assist with Python code drafting, feature-engineering ideas,
debugging, visualisations and report wording. The assistant executed the notebook
and checked the recorded outputs. I need to verify the final code, interpretations
and conclusions before submission. This declaration describes the assistance
actually used and does not claim that AI-drafted work was unaided original work.

## Aim and scope
Continue Portfolio 1 using the UCI Concrete Compressive Strength dataset. Build
four relative strength classes; compare five decision-tree feature sets; then
compare an SVM baseline, tuned SVM, K-Best SVM and PCA SVM. Report measured results
without assuming that every transformation improves performance.

The modelling workflow is self-contained. Run all cells in order from this
notebook's folder; the original dataset is included and execution needs no
network connection.
''')
md('''# 1. Target preparation and evaluation design
## Connection to Portfolio 1
Portfolio 1 Tasks 2-3 found no missing values, 25 extra exact duplicate records and
a broad strength distribution. Task 4 found strongly right-skewed age and optional
ingredients with many zeros. Tasks 5-6 identified cement, superplasticizer and age
as promising signals and suggested water-to-binder ratios and age transformations.
These findings motivate the choices below; all fitted statistics are recalculated
on the relevant training data, not imported from the full-data EDA.

The original target is continuous strength in MPa. This portfolio uses four ordered
classification labels: Lower, Lower-middle, Upper-middle and Higher. Quartiles
produce approximately balanced training classes. They are sample-relative labels,
not structural grades or safety thresholds. Target-derived quantities never enter X.
''')
code('''
from pathlib import Path
import json
import hashlib
import platform
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.metrics import (accuracy_score, f1_score, balanced_accuracy_score,
                             confusion_matrix, classification_report)

SEED = 42
CLASS_NAMES = ['Lower', 'Lower-middle', 'Upper-middle', 'Higher']
CLASSES = np.arange(4)
ORIGINAL = ['cement', 'slag', 'fly_ash', 'water', 'superplasticizer',
            'coarse_aggregate', 'fine_aggregate', 'age']
INGREDIENTS = ORIGINAL[:-1]
for folder in ['figures', 'results']:
    Path(folder).mkdir(exist_ok=True)
pd.set_option('display.width', 110)
pd.set_option('display.max_columns', 14)
pd.set_option('display.precision', 4)
sns.set_theme(style='whitegrid', font_scale=0.9)

def table(data):
    print(data.to_string())

def figure(name):
    plt.savefig(Path('figures') / f'{name}.png', dpi=180, bbox_inches='tight')
    plt.show()
    plt.close('all')

source = Path('data/Concrete_Data.xls')
raw = pd.read_excel(source, engine='xlrd')
assert raw.shape == (1030, 9)
raw.columns = ORIGINAL + ['strength']
assert raw.notna().all().all() and np.isfinite(raw.to_numpy()).all()
duplicates = int(raw.duplicated().sum())
data = raw.drop_duplicates().copy()
data['age'] = data.age.astype(int)
X = data[ORIGINAL].copy()
y = data.strength.copy()
# Same seven ingredients = same recipe, even if age or measured strength differs.
groups, recipe_values = pd.factorize(pd.MultiIndex.from_frame(X[INGREDIENTS]))
splitter = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
train_pos, test_pos = next(splitter.split(X, y, groups))
X_train, X_test = X.iloc[train_pos], X.iloc[test_pos]
y_train, y_test = y.iloc[train_pos], y.iloc[test_pos]
g_train, g_test = groups[train_pos], groups[test_pos]
assert set(g_train).isdisjoint(g_test)
assert X_train.index.intersection(X_test.index).empty
print('Python:', platform.python_version(), '| scikit-learn:', sklearn.__version__)
print('Source SHA-256:', hashlib.sha256(source.read_bytes()).hexdigest())
print('Raw shape:', raw.shape, '| exact duplicates removed:', duplicates)
print('Clean shape:', data.shape, '| distinct recipes:', len(recipe_values))
print(f'Train: {len(X_train)} rows ({len(X_train)/len(data):.1%}), '
      f'{len(set(g_train))} recipes')
print(f'Test:  {len(X_test)} rows ({len(X_test)/len(data):.1%}), '
      f'{len(set(g_test))} recipes')
print('Shared recipes across train/test:', len(set(g_train) & set(g_test)))
table(raw.isna().sum().to_frame('missing_count'))
''')
md('''## Why split by recipe?
A mixture can be measured at several ages. A random row split could let the model
see the same mixture in training and testing. The chosen split reserves 30% of
recipe groups, so the row ratio is approximately rather than exactly 70/30. This
evaluates performance on unseen recipes within this dataset, not new laboratories.
The group identifier uses the ingredient measurements exactly as supplied; near-
identical recipes and unknown batches may still be related.

Exact duplicate removal follows Portfolio 1. Without specimen IDs, repetitions
could be either repeated tests or duplicate entries, so the original workbook is
preserved. Missing values are absent, optional-ingredient zeros are retained, and
unusual but plausible values are not deleted.
''')
code('''
def learn_edges(values):
    edges = np.quantile(np.asarray(values), [0.25, 0.50, 0.75])
    if not np.all(np.diff(edges) > 0):
        raise ValueError('Quartile boundaries are not distinct.')
    return edges

def encode_strength(values, edges):
    # Class 0 <= q25, class 1 (q25,q50], class 2 (q50,q75], class 3 > q75.
    # Infinite outer intervals handle future strengths outside training extrema.
    return np.searchsorted(edges, np.asarray(values), side='left').astype(int)

train_edges = learn_edges(y_train)
y_train_class = encode_strength(y_train, train_edges)
y_test_class = encode_strength(y_test, train_edges)
class_counts = pd.DataFrame({
    'train_count': np.bincount(y_train_class, minlength=4),
    'test_count': np.bincount(y_test_class, minlength=4)}, index=CLASS_NAMES)
class_counts['train_percent'] = 100 * class_counts.train_count / len(y_train)
class_counts['test_percent'] = 100 * class_counts.test_count / len(y_test)
print('Training-only boundaries (MPa):', train_edges.tolist())
print('Rules: <=q25; (q25,q50]; (q50,q75]; >q75')
table(class_counts.round(3))
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
sns.histplot(y_train, bins=25, ax=axes[0], color='#28749b')
for edge in train_edges:
    axes[0].axvline(edge, color='#bd5739', linestyle='--')
axes[0].set(title='Figure 1a. Training strength with quartiles',
            xlabel='Compressive strength (MPa)', ylabel='Training observations')
class_counts[['train_percent', 'test_percent']].plot.bar(ax=axes[1], rot=18)
axes[1].set(title='Figure 1b. Class balance using frozen train edges',
            xlabel='Relative strength class', ylabel='Percentage within partition')
axes[1].legend(['Train', 'Test'])
plt.tight_layout()
figure('01_target_classes')
''')
md('''## Cross-validation and target handling
All comparisons use five group folds on the training partition. In each fit, the
estimator learns quartile boundaries only from that fold's continuous training
strengths, then classifies its validation strengths using those frozen boundaries.
Thus neither holdout labels nor validation labels determine their own thresholds.
The four label meanings are relative to each training sample, so boundaries can
vary slightly between folds. CV measures the complete learning procedure, including
target preparation, rather than performance against one globally pre-binned target.

Accuracy is easy to read; macro F1 weights each class equally and is the primary
selection metric. Balanced accuracy checks class-wise recall. Mean absolute class
error additionally distinguishes a one-band mistake from a three-band mistake.
These are classification metrics, not MPa prediction errors.
''')
md('''# 2. Normalisation and encoding
Portfolio 1 Task 4 showed near-symmetric water and aggregate distributions, but
asymmetric cement, zero-heavy optional ingredients and strongly skewed age. Use
StandardScaler for water/coarse aggregate/fine aggregate, and RobustScaler for
cement/slag/fly ash/superplasticizer/age. These choices are documented per feature
and checked on training data below. Robust scaling reduces the influence of extremes
on the scale estimate; it does not remove outliers or make a distribution normal.

All eight predictors are numeric, so no predictor one-hot or label encoding is
needed. Only the ordered target labels are represented by integers 0-3. SVC treats
these as class identifiers, not a continuous response, so it does not explicitly
optimise ordinal distance. Every scaler is fitted inside its training pipeline.
''')
code('''
SYMMETRIC = ['water', 'coarse_aggregate', 'fine_aggregate']
ROBUST = [c for c in ORIGINAL if c not in SYMMETRIC]
scale_reasons = {
    'cement': 'Broad asymmetric binder distribution; median/IQR scaling',
    'slag': 'Zero-heavy optional ingredient; broad positive range',
    'fly_ash': 'Zero-heavy optional ingredient; non-Gaussian mixture',
    'water': 'Near-symmetric, concentrated distribution; mean/SD scaling',
    'superplasticizer': 'Many zeros and positive extremes; median/IQR scaling',
    'coarse_aggregate': 'Near-symmetric distribution; mean/SD scaling',
    'fine_aggregate': 'Approximately symmetric compared with age/admixtures',
    'age': 'Strong right tail and long curing ages; median/IQR scaling'}
audit = pd.DataFrame({'skewness': X_train.skew(),
                       'zero_percent': 100 * X_train.eq(0).mean(),
                       'scaler': ['Standard' if c in SYMMETRIC else 'Robust'
                                  for c in ORIGINAL]}, index=ORIGINAL)
table(audit.round(3))
for c in ORIGINAL:
    print(c + ': ' + scale_reasons[c])

def make_scaler(names):
    standard = [c for c in names if c in SYMMETRIC]
    robust = [c for c in names if c not in SYMMETRIC]
    parts = []
    if standard:
        parts.append(('standard', StandardScaler(), standard))
    if robust:
        parts.append(('robust', RobustScaler(), robust))
    return ColumnTransformer(parts, verbose_feature_names_out=False)

demo_scaler = make_scaler(ORIGINAL).fit(X_train)
scaled_names = demo_scaler.get_feature_names_out()
scaled_train = pd.DataFrame(demo_scaler.transform(X_train),
                            columns=scaled_names, index=X_train.index)[ORIGINAL]
comparison = pd.DataFrame({'before_median': X_train.median(),
    'before_IQR': X_train.quantile(.75) - X_train.quantile(.25),
    'after_median': scaled_train.median(),
    'after_IQR': scaled_train.quantile(.75) - scaled_train.quantile(.25)})
table(comparison.round(3))
fig, axes = plt.subplots(2, 1, figsize=(11, 6))
sns.boxplot(data=X_train, ax=axes[0], color='#91b8ca', fliersize=2)
sns.boxplot(data=scaled_train, ax=axes[1], color='#72ad97', fliersize=2)
axes[0].set(title='Figure 2a. Original training predictor scales',
            xlabel='', ylabel='Raw value (mixed units)')
axes[1].set(title='Figure 2b. Training predictors after column-wise scaling',
            xlabel='Predictor', ylabel='Scaled value (dimensionless)')
for ax in axes:
    ax.tick_params(axis='x', rotation=20)
plt.tight_layout()
figure('02_scaling_comparison')
''')
md('''Scaling matters for an RBF SVM because feature distances determine similarity.
Without it, ingredient quantities in hundreds of kg/m^3 could dominate dosage or
ratio features. A separate comparison of scaled and unscaled SVM performance below
tests this effect. Axis-aligned decision trees usually do not benefit from monotonic
scaling, but using the same preprocessing makes the comparison consistent. PCA is
sensitive to scale; the PCA-specific pipeline uses an additional StandardScaler so
all inputs have unit variance before components are calculated.
''')
md('''# 3. Feature engineering
Five deterministic features extend the eight original predictors:

- `binder_total = cement + slag + fly_ash` (kg/m^3): represents total nominal binder;
  motivated by the mixed ingredient relationships in Portfolio 1 Tasks 5-6.
- `water_binder_ratio = water / binder_total` (dimensionless): compares water to
  nominal binder, following the negative water-strength association in Task 5.
- `log_age = log(1 + age)` (log of day count): addresses the strong age skew in Task 4
  and the nonlinear age-strength hypothesis in Task 6.
- `scm_fraction = (slag + fly_ash) / binder_total` (dimensionless): captures binder
  replacement rather than treating weak marginal correlations as irrelevance.
- `maturity_ratio = log_age / water_binder_ratio` (dimensionless numerical proxy):
  explores an age-water interaction suggested by Task 6. It is an experimental
  composite, not a validated maturity index or physical strength law.

All ingredients contribute equally to the nominal binder sum for bookkeeping; this
does not assert equal reactivity. Positive cement and water make denominators valid
here. No feature uses measured strength or the class label. Correlations below are
training-only descriptions, not proof of causal influence or predictive improvement.
''')
code('''
ENGINEERED = ['binder_total', 'water_binder_ratio', 'log_age',
              'scm_fraction', 'maturity_ratio']
ALL_FEATURES = ORIGINAL + ENGINEERED
DOMAIN = ['binder_total', 'water_binder_ratio', 'log_age',
          'scm_fraction', 'superplasticizer']

def engineer(frame):
    out = frame[ORIGINAL].copy()
    out['binder_total'] = out.cement + out.slag + out.fly_ash
    if (out.binder_total <= 0).any() or (out.water <= 0).any():
        raise ValueError('Positive binder and water are required for ratios.')
    out['water_binder_ratio'] = out.water / out.binder_total
    out['log_age'] = np.log1p(out.age)
    out['scm_fraction'] = (out.slag + out.fly_ash) / out.binder_total
    out['maturity_ratio'] = out.log_age / out.water_binder_ratio
    assert np.isfinite(out.to_numpy()).all()
    assert out.scm_fraction.between(0, 1).all()
    return out

engineered_train = engineer(X_train)
association = pd.DataFrame({
    'Pearson_r': engineered_train.corrwith(y_train),
    'Spearman_rho': engineered_train.corrwith(y_train, method='spearman')})
table(association.loc[ENGINEERED].round(4))
print('Engineered feature validity checks passed.')
print('Strongest pairwise dependencies among expanded inputs:')
corr = engineered_train.corr()
pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(ALL_FEATURES)
         for b in ALL_FEATURES[i+1:] if abs(corr.loc[a, b]) >= .80]
table(pd.DataFrame(pairs, columns=['feature_1', 'feature_2', 'Pearson_r']).round(4))
fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
for c, ax in zip(['water_binder_ratio', 'log_age', 'maturity_ratio'], axes):
    ax.scatter(engineered_train[c], y_train, alpha=.35, s=12, color='#28749b')
    ax.set(xlabel=c.replace('_', ' '), ylabel='Strength (MPa)')
fig.suptitle('Figure 3. Engineered features versus training strength', y=1.02)
plt.tight_layout()
figure('03_engineered_relationships')
''')
md('''Aggregates and ingredients are constrained mixture components; derived ratios
and interactions also share information with their source columns. High dependence
is therefore expected. Feature engineering can help a shallow tree represent a
useful relationship with fewer splits, but adding redundant dimensions can hurt an
SVM. The original-versus-expanded feature comparison tests the net effect; no
feature is declared useful solely because it correlates with strength.
''')
md('''# 4. Decision-tree modelling across five feature sets
The five sets are original eight predictors, five strongest absolute training
Pearson correlations with continuous strength, all thirteen original plus engineered
features, five PCA components of scaled original features, and five domain-selected
features. Top-correlation ranking and PCA are refitted within every training fold.
The PCA tree deliberately uses five components as a comparison condition; its
variance retention is inspected later rather than assumed optimal.

All five trees use `max_depth=6`, `min_samples_leaf=5` and seed 42. Keeping these
settings fixed isolates the feature-set comparison and limits uncontrolled tree
growth. This is not a claim that these are the best tree hyperparameters.
''')
code('''
class FeatureBuilder(TransformerMixin, BaseEstimator):
    def __init__(self, feature_set='original'):
        self.feature_set = feature_set

    def fit(self, X, y=None):
        if self.feature_set == 'top5':
            scores = X[ORIGINAL].corrwith(pd.Series(np.asarray(y), index=X.index))
            self.columns_ = list(scores.abs().sort_values(ascending=False).head(5).index)
        elif self.feature_set == 'expanded':
            self.columns_ = ALL_FEATURES.copy()
        elif self.feature_set == 'domain':
            self.columns_ = DOMAIN.copy()
        else:
            self.columns_ = ORIGINAL.copy()
        self.feature_names_in_ = np.asarray(ORIGINAL, dtype=object)
        return self

    def transform(self, X):
        return engineer(X)[self.columns_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns_, dtype=object)

class AdaptiveScaler(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.scaler_ = make_scaler(list(X.columns)).fit(X)
        return self

    def transform(self, X):
        return self.scaler_.transform(X)

    def get_feature_names_out(self, input_features=None):
        return self.scaler_.get_feature_names_out()

def preprocessing(feature_set='expanded', reduction=None, unscaled=False):
    steps = [('build', FeatureBuilder(feature_set))]
    if not unscaled:
        steps.append(('scale', AdaptiveScaler()))
    if reduction == 'pca5':
        steps += [('unit_variance', StandardScaler()), ('pca', PCA(n_components=5))]
    elif reduction == 'pca':
        steps += [('unit_variance', StandardScaler()),
                  ('pca', PCA(n_components=.90, svd_solver='full'))]
    elif reduction == 'kbest':
        steps.append(('kbest', SelectKBest(score_func=f_classif, k=8)))
    return Pipeline(steps)

class RelativeStrengthClassifier(ClassifierMixin, BaseEstimator):
    # y remains continuous at the API boundary so binning can be learned per fit.
    def __init__(self, features=None, classifier=None, selection_uses_classes=False):
        self.features = features
        self.classifier = classifier
        self.selection_uses_classes = selection_uses_classes

    def fit(self, X, y):
        self.edges_ = learn_edges(y)
        labels = self.encode_target(y)
        self.classes_ = CLASSES.copy()
        selection_y = labels if self.selection_uses_classes else np.asarray(y)
        self.features_ = clone(self.features)
        transformed = self.features_.fit_transform(X, selection_y)
        self.classifier_ = clone(self.classifier).fit(transformed, labels)
        self.n_features_in_ = X.shape[1]
        return self

    def encode_target(self, y):
        return encode_strength(y, self.edges_)

    def predict(self, X):
        return self.classifier_.predict(self.features_.transform(X))

    def score(self, X, y):
        return accuracy_score(self.encode_target(y), self.predict(X))

def metrics(estimator, X, raw_y):
    actual, predicted = estimator.encode_target(raw_y), estimator.predict(X)
    return {'accuracy': accuracy_score(actual, predicted),
            'macro_f1': f1_score(actual, predicted, labels=CLASSES,
                                  average='macro', zero_division=0),
            'balanced_accuracy': balanced_accuracy_score(actual, predicted),
            'class_MAE': float(np.abs(actual - predicted).mean())}

def search_scoring(estimator, X, raw_y):
    # A callable scorer bins validation strengths using fitted training edges.
    result = metrics(estimator, X, raw_y)
    return {k: v for k, v in result.items() if k != 'class_MAE'}

outer_cv = GroupKFold(n_splits=5, shuffle=True, random_state=SEED)
FOLDS = list(outer_cv.split(X_train, y_train, g_train))
for fit_idx, valid_idx in FOLDS:
    assert set(g_train[fit_idx]).isdisjoint(g_train[valid_idx])
print('Five recipe-group CV folds verified; all target preparation is inside fit().')
''')
code('''
def compare_on_training(name, estimator, parameter_grid=None):
    start = time.perf_counter()
    rows = []
    for fold, (fit_idx, valid_idx) in enumerate(FOLDS, start=1):
        candidate = clone(estimator)
        if parameter_grid is not None:
            candidate = GridSearchCV(candidate, parameter_grid,
                scoring=search_scoring, refit='macro_f1',
                cv=GroupKFold(n_splits=5, shuffle=True, random_state=SEED + fold),
                n_jobs=1, error_score='raise')
            candidate.fit(X_train.iloc[fit_idx], y_train.iloc[fit_idx],
                          groups=g_train[fit_idx])
            fitted = candidate.best_estimator_
        else:
            fitted = candidate.fit(X_train.iloc[fit_idx], y_train.iloc[fit_idx])
        score = metrics(fitted, X_train.iloc[valid_idx], y_train.iloc[valid_idx])
        score.update(fold=fold, model=name,
                     q25=fitted.edges_[0], q50=fitted.edges_[1], q75=fitted.edges_[2])
        rows.append(score)
    if parameter_grid is None:
        final = clone(estimator).fit(X_train, y_train)
        search = None
    else:
        search = GridSearchCV(clone(estimator), parameter_grid,
            scoring=search_scoring, refit='macro_f1',
            cv=GroupKFold(n_splits=5, shuffle=True, random_state=SEED + 100),
            n_jobs=1, error_score='raise')
        search.fit(X_train, y_train, groups=g_train)
        final = search.best_estimator_
    folds = pd.DataFrame(rows)
    summary = {'model': name, 'features': final.features_.transform(X_train).shape[1],
               'train_accuracy': metrics(final, X_train, y_train)['accuracy'],
               'CV_accuracy': folds.accuracy.mean(),
               'CV_accuracy_SD': folds.accuracy.std(ddof=1),
               'CV_macro_F1': folds.macro_f1.mean(),
               'CV_F1_SD': folds.macro_f1.std(ddof=1),
               'CV_class_MAE': folds.class_MAE.mean(),
               'seconds': time.perf_counter() - start}
    print(f'{name}: 5-fold macro F1={summary["CV_macro_F1"]:.4f} '
          f'+/- {summary["CV_F1_SD"]:.4f}', flush=True)
    return summary, final, folds, search

TREE_SETS = {'DT Original': ('original', None),
             'DT Top 5': ('top5', None),
             'DT Expanded': ('expanded', None),
             'DT PCA 5': ('original', 'pca5'),
             'DT Domain': ('domain', None)}
tree_rows, tree_models, fold_records = [], {}, []
for name, (feature_set, reduction) in TREE_SETS.items():
    estimator = RelativeStrengthClassifier(
        features=preprocessing(feature_set, reduction),
        classifier=DecisionTreeClassifier(max_depth=6, min_samples_leaf=5,
                                            random_state=SEED))
    summary, fitted, folds, _ = compare_on_training(name, estimator)
    tree_rows.append(summary)
    tree_models[name] = fitted
    fold_records.append(folds)
tree_cv = pd.DataFrame(tree_rows).set_index('model')
best_tree_name = tree_cv.CV_macro_F1.idxmax()
best_tree = tree_models[best_tree_name]
print('Best tree by training CV macro F1:', best_tree_name)
print('Full-training top-correlated features:',
      tree_models['DT Top 5'].features_.named_steps['build'].columns_)
''')
md('''# 5. First comparison table and interpretation
All five feature sets use the same holdout and the same five outer folds. The tree
winner is chosen by training CV macro F1 before its holdout score is inspected.
The table reports both 70/30-style recipe holdout evaluation and CV estimates.
Holdout scores are comparison evidence; they do not choose features or parameters.
''')
code('''
tree_holdout = pd.DataFrame({name: metrics(model, X_test, y_test)
                            for name, model in tree_models.items()}).T
first_table = tree_cv.join(tree_holdout.add_prefix('test_'))
first_columns = ['features', 'train_accuracy', 'CV_accuracy', 'CV_macro_F1',
                 'CV_F1_SD', 'test_accuracy', 'test_macro_f1']
table(first_table[first_columns].round(4))
first_table.to_csv('results/first_decision_tree_comparison.csv')
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
positions = np.arange(len(tree_cv))
axes[0].bar(positions - .18, tree_cv.CV_macro_F1, .36, label='Group CV mean')
axes[0].bar(positions + .18, tree_holdout.macro_f1, .36, label='Recipe holdout')
axes[0].errorbar(positions - .18, tree_cv.CV_macro_F1,
                 yerr=tree_cv.CV_F1_SD, fmt='none', color='black', capsize=3)
axes[0].set(xticks=positions, xticklabels=tree_cv.index, ylim=(0, 1),
            title='Figure 4a. Five decision-tree feature sets', ylabel='Macro F1')
axes[0].tick_params(axis='x', rotation=25)
axes[0].legend(fontsize=8)
importance = pd.Series(best_tree.classifier_.feature_importances_,
    index=best_tree.features_.get_feature_names_out()).sort_values()
importance.plot.barh(ax=axes[1], color='#438573')
axes[1].set(title=f'Figure 4b. {best_tree_name}: impurity importance',
            xlabel='Training impurity-based importance (sum = 1)', ylabel='Model input')
plt.tight_layout()
figure('04_tree_comparison_importance')
print('Best-tree input importances:')
table(importance.sort_values(ascending=False).to_frame('importance').round(4))
''')
md('''The tree table tests whether a smaller set retains useful information, whether
domain features improve on raw inputs, and whether PCA helps an axis-aligned tree.
The winning feature set and measured differences are summarised in Section 10.
Training-versus-CV gaps indicate possible overfitting. Fold standard deviations
describe variability across these five splits; they are not confidence intervals.
Impurity importance explains how this fitted tree used its inputs, not causal
importance. Correlated features can share or substitute for each other; PCA inputs,
if selected, must be interpreted as components rather than original ingredients.
''')
md('''# 6. SVM baseline and five-fold cross-validation
The four-class task calls for `SVC`, not SVR. The baseline uses all thirteen inputs
with the documented scaling, an RBF kernel, C=1 and gamma='scale'. A separate
unscaled SVC is a diagnostic control rather than a fifth final model variation.
This makes the effect of scaling measurable rather than assumed.
''')
code('''
svm_base = RelativeStrengthClassifier(features=preprocessing('expanded'),
    classifier=SVC(kernel='rbf', C=1, gamma='scale', cache_size=512))
baseline_summary, baseline_svm, baseline_folds, _ = compare_on_training(
    'SVM Baseline', svm_base)
svm_rows = [baseline_summary]
svm_models = {'SVM Baseline': baseline_svm}
fold_records.append(baseline_folds)
unscaled = RelativeStrengthClassifier(
    features=preprocessing('expanded', unscaled=True),
    classifier=SVC(kernel='rbf', C=1, gamma='scale', cache_size=512))
unscaled_summary, _, _, _ = compare_on_training('SVM Unscaled control', unscaled)
table(pd.DataFrame([unscaled_summary, baseline_summary])[
    ['model', 'CV_accuracy', 'CV_macro_F1', 'CV_F1_SD']].set_index('model').round(4))
print('Baseline holdout metrics (fixed model; no tuning from holdout):')
table(pd.Series(metrics(baseline_svm, X_test, y_test)).to_frame('value').round(4))
''')
md('''# 7. Hyperparameter tuning
GridSearchCV compares C in {1, 10, 100} and gamma in {'scale', 0.03, 0.1}, giving
nine RBF-SVC configurations. C controls the regularisation-error trade-off; gamma
controls the locality of the RBF response. The grid is intentionally bounded for
reproducibility, not claimed to exhaust plausible settings.

Five-fold inner group CV selects parameters using macro F1. Five outer group folds
estimate the performance of this tuning procedure. A final inner search on all
training recipes chooses the model that will be evaluated on the holdout. Its
`best_score_` is a selection score, not an unbiased performance estimate. Nested
outer CV and the reserved holdout provide separate evaluation evidence.
''')
code('''
svm_grid = {'classifier__C': [1, 10, 100],
            'classifier__gamma': ['scale', .03, .1]}
tuned_summary, tuned_svm, tuned_folds, tuned_search = compare_on_training(
    'SVM Tuned', svm_base, svm_grid)
svm_rows.append(tuned_summary)
svm_models['SVM Tuned'] = tuned_svm
fold_records.append(tuned_folds)
print('Full-training search best parameters:', tuned_search.best_params_)
print('Inner selection macro F1:', round(tuned_search.best_score_, 4))
print('Nested outer macro F1:', round(tuned_summary['CV_macro_F1'], 4))
tuning_results = pd.DataFrame(tuned_search.cv_results_)
table(tuning_results[['param_classifier__C', 'param_classifier__gamma',
    'mean_test_macro_f1', 'std_test_macro_f1', 'rank_test_macro_f1']]
    .sort_values('rank_test_macro_f1').round(4))
tuning_results.to_csv('results/svm_tuning_grid.csv', index=False)
''')
md('''# 8. K-Best feature selection
SelectKBest with ANOVA F scores evaluates class-conditional mean separation.
It receives class labels formed from each fit's training quartiles, not continuous
strength values. Candidate K values {5, 8, 11} test compact, intermediate and broad
subsets of the thirteen features. Joint inner CV selects K, C and gamma; therefore
the selected K is supported by training validation, not chosen from holdout results.
The univariate F scores do not measure interactions or causality and have no role
as formal significance tests for these dependent, grouped observations.
''')
code('''
kbest_base = RelativeStrengthClassifier(
    features=preprocessing('expanded', 'kbest'),
    classifier=SVC(kernel='rbf', cache_size=512), selection_uses_classes=True)
kbest_grid = {'features__kbest__k': [5, 8, 11],
              'classifier__C': [1, 10], 'classifier__gamma': ['scale', .1]}
k_summary, k_svm, k_folds, k_search = compare_on_training(
    'SVM K-Best', kbest_base, kbest_grid)
svm_rows.append(k_summary)
svm_models['SVM K-Best'] = k_svm
fold_records.append(k_folds)
print('Best K-Best parameters:', k_search.best_params_)
print('Selected features:', list(k_svm.features_.get_feature_names_out()))
pre_select_names = k_svm.features_[:-1].get_feature_names_out()
f_scores = pd.Series(k_svm.features_.named_steps['kbest'].scores_,
                     index=pre_select_names).sort_values(ascending=False)
table(f_scores.to_frame('training_ANOVA_F').round(3))
''')
md('''# 9. Dimensionality reduction with PCA
PCA is unsupervised: it retains directions with high input variance, which need
not be the directions that best separate strength classes. After the ordinary
column-wise scaling, a PCA-specific StandardScaler gives every input unit variance.
The candidate retained-variance thresholds are 80%, 90% and 95%. Inner CV jointly
selects the threshold, C and gamma. The actual number of components depends on each
training fold; the reported count describes the final full-training fit.

Both selection and PCA stay inside the estimator passed to every inner and outer
fold. Thus validation and holdout inputs cannot influence the component directions,
feature ranking or fitted scale values.
''')
code('''
pca_base = RelativeStrengthClassifier(features=preprocessing('expanded', 'pca'),
    classifier=SVC(kernel='rbf', cache_size=512))
pca_grid = {'features__pca__n_components': [.80, .90, .95],
            'classifier__C': [1, 10], 'classifier__gamma': ['scale', .1]}
p_summary, p_svm, p_folds, p_search = compare_on_training(
    'SVM PCA', pca_base, pca_grid)
svm_rows.append(p_summary)
svm_models['SVM PCA'] = p_svm
fold_records.append(p_folds)
print('Best PCA parameters:', p_search.best_params_)
chosen_pca = p_svm.features_.named_steps['pca']
print('Selected components:', chosen_pca.n_components_)
print('Retained variance:', round(chosen_pca.explained_variance_ratio_.sum(), 4))
before_pca = p_svm.features_[:-1].transform(X_train)
pca_diagnostic = PCA().fit(before_pca)
variance = pd.DataFrame({'component': np.arange(1, before_pca.shape[1] + 1),
    'variance_ratio': pca_diagnostic.explained_variance_ratio_,
    'cumulative': np.cumsum(pca_diagnostic.explained_variance_ratio_)})
table(variance.round(4))
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].bar(variance.component, variance.variance_ratio, color='#28749b')
axes[0].set(title='Figure 5a. Training PCA explained variance',
            xlabel='Principal component', ylabel='Explained variance fraction')
axes[1].plot(variance.component, variance.cumulative, marker='o')
axes[1].axvline(chosen_pca.n_components_, color='#bd5739', linestyle='--',
                label=f'Chosen: {chosen_pca.n_components_} components')
axes[1].axhline(float(p_search.best_params_['features__pca__n_components']),
                color='#438573', linestyle=':', label='Selected variance threshold')
axes[1].set(title='Figure 5b. Cumulative training variance',
            xlabel='Number of components', ylabel='Cumulative explained fraction',
            ylim=(0, 1.05))
axes[1].legend(fontsize=8)
plt.tight_layout()
figure('05_pca_variance')
''')
md('''# 10. Final comparison, interpretation and limitations
The final table contains the four SVM variations. Baseline CV evaluates a fixed
recipe; the other rows report nested CV that repeats their complete searches.
Their CV estimates therefore reflect their respective learning procedures. All
holdout scores use the same recipe partition and the same full-training quartile
boundaries, making the holdout classes directly comparable across fitted models.

The recommended SVM is selected using CV macro F1 before the final holdout table
is created. All variants were specified in advance; no additional tuning is done
after inspecting holdout performance. Reporting multiple models on one holdout
still calls for caution: further iteration based on these scores would require a
new validation dataset.
''')
code('''
svm_cv = pd.DataFrame(svm_rows).set_index('model')
selected_svm_name = svm_cv.CV_macro_F1.idxmax()
selected_svm = svm_models[selected_svm_name]
svm_holdout = pd.DataFrame({name: metrics(model, X_test, y_test)
                           for name, model in svm_models.items()}).T
final_table = svm_cv.join(svm_holdout.add_prefix('test_'))
final_columns = ['features', 'CV_accuracy', 'CV_accuracy_SD', 'CV_macro_F1',
                 'CV_F1_SD', 'test_accuracy', 'test_macro_f1', 'test_class_MAE']
table(final_table[final_columns].round(4))
print('SVM selected using training CV:', selected_svm_name)
print('Tuned-minus-baseline outer CV F1:',
      round(svm_cv.loc['SVM Tuned', 'CV_macro_F1'] - baseline_summary['CV_macro_F1'], 4))
print('Tuned-minus-baseline holdout F1:',
      round(svm_holdout.loc['SVM Tuned', 'macro_f1'] -
            svm_holdout.loc['SVM Baseline', 'macro_f1'], 4))
print('Scaling-control CV F1 difference:',
      round(baseline_summary['CV_macro_F1'] - unscaled_summary['CV_macro_F1'], 4))
majority_class = int(np.argmax(np.bincount(y_train_class, minlength=4)))
dummy_pred = np.repeat(majority_class, len(y_test_class))
print('Training-majority dummy holdout accuracy:',
      round(accuracy_score(y_test_class, dummy_pred), 4))
print('Selected SVM holdout class report:')
print(classification_report(y_test_class, selected_svm.predict(X_test),
    labels=CLASSES, target_names=CLASS_NAMES, digits=3, zero_division=0))
final_table.to_csv('results/final_svm_comparison.csv')
all_folds = pd.concat(fold_records, ignore_index=True)
all_folds.to_csv('results/cross_validation_folds.csv', index=False)
''')
code('''
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
positions = np.arange(len(svm_cv))
axes[0].bar(positions - .18, svm_cv.CV_macro_F1, .36, label='Outer group CV mean')
axes[0].bar(positions + .18, svm_holdout.macro_f1, .36, label='Recipe holdout')
axes[0].errorbar(positions - .18, svm_cv.CV_macro_F1, yerr=svm_cv.CV_F1_SD,
                 fmt='none', capsize=3, color='black')
axes[0].set(xticks=positions, xticklabels=svm_cv.index, ylim=(0, 1),
            title='Figure 6a. Four SVM variations', ylabel='Macro F1')
axes[0].tick_params(axis='x', rotation=20)
axes[0].legend(fontsize=8)
matrix = confusion_matrix(y_test_class, selected_svm.predict(X_test), labels=CLASSES)
sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=axes[1],
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
            cbar_kws={'label': 'Holdout observations'})
axes[1].set(title=f'Figure 6b. {selected_svm_name}: holdout confusion matrix',
            xlabel='Predicted relative strength class', ylabel='Actual class')
axes[1].tick_params(axis='x', rotation=20)
plt.tight_layout()
figure('06_final_svm_comparison')
''')
code('''
# Save an audit trail; no executable model pickle is needed to reproduce this work.
split_audit = data.copy()
split_audit['recipe_group'] = groups
split_audit['partition'] = 'test'
split_audit.loc[X_train.index, 'partition'] = 'train'
split_audit['strength_class_code'] = encode_strength(data.strength, train_edges)
split_audit['strength_class'] = np.asarray(CLASS_NAMES)[split_audit.strength_class_code]
split_audit.to_csv('data/partitioned_concrete.csv', index_label='source_row')
predictions = pd.DataFrame({'actual_strength_MPa': y_test,
    'actual_class_code': y_test_class}, index=X_test.index)
for name, model in {**tree_models, **svm_models}.items():
    assert np.allclose(model.edges_, train_edges)
    predictions[name] = model.predict(X_test)
predictions.to_csv('results/holdout_predictions.csv', index_label='source_row')
pd.DataFrame({'feature': list(scale_reasons),
              'reason': list(scale_reasons.values())}).to_csv(
                  'results/scaling_rationale.csv', index=False)
metadata = {
    'author': 'Pham Quang Minh', 'student_id': '104999568',
    'date': '2026-10-18', 'seed': SEED,
    'source_doi': '10.24432/C5PK67',
    'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
    'raw_rows': len(raw), 'clean_rows': len(data), 'duplicates_removed': duplicates,
    'train_rows': len(X_train), 'test_rows': len(X_test),
    'train_recipes': len(set(g_train)), 'test_recipes': len(set(g_test)),
    'class_names': CLASS_NAMES, 'training_edges_MPa': train_edges.tolist(),
    'class_rule': '(-inf,q25], (q25,q50], (q50,q75], (q75,inf)',
    'target_CV': 'Quartiles refitted within each fit, including inner and outer folds',
    'best_tree_by_CV': best_tree_name, 'selected_SVM_by_CV': selected_svm_name,
    'selected_K': int(k_search.best_params_['features__kbest__k']),
    'K_features': list(k_svm.features_.get_feature_names_out()),
    'PCA_components': int(chosen_pca.n_components_),
    'PCA_variance': float(chosen_pca.explained_variance_ratio_.sum()),
    'tuned_parameters': tuned_search.best_params_,
    'K_parameters': k_search.best_params_, 'PCA_parameters': p_search.best_params_,
    'versions': {'python': platform.python_version(), 'sklearn': sklearn.__version__,
                 'numpy': np.__version__, 'pandas': pd.__version__}}
Path('results/reproducibility.json').write_text(json.dumps(metadata, indent=2))
summary = {
    'best_tree': best_tree_name,
    'tree_CV_F1': float(tree_cv.loc[best_tree_name, 'CV_macro_F1']),
    'tree_test_F1': float(tree_holdout.loc[best_tree_name, 'macro_f1']),
    'best_svm': selected_svm_name,
    'svm_CV_F1': float(svm_cv.loc[selected_svm_name, 'CV_macro_F1']),
    'svm_test_F1': float(svm_holdout.loc[selected_svm_name, 'macro_f1']),
    'svm_test_accuracy': float(svm_holdout.loc[selected_svm_name, 'accuracy']),
    'svm_test_class_MAE': float(svm_holdout.loc[selected_svm_name, 'class_MAE']),
    'tuning_CV_delta': float(tuned_summary['CV_macro_F1'] -
                             baseline_summary['CV_macro_F1']),
    'tuning_test_delta': float(svm_holdout.loc['SVM Tuned', 'macro_f1'] -
                               svm_holdout.loc['SVM Baseline', 'macro_f1']),
    'scale_CV_delta': float(baseline_summary['CV_macro_F1'] -
                            unscaled_summary['CV_macro_F1']),
    'expanded_tree_delta': float(tree_cv.loc['DT Expanded', 'CV_macro_F1'] -
                                  tree_cv.loc['DT Original', 'CV_macro_F1']),
    'dummy_accuracy': float(accuracy_score(y_test_class, dummy_pred))}
Path('results/summary.json').write_text(json.dumps(summary, indent=2))
assert len(tree_models) == 5 and len(svm_models) == 4
assert len(all_folds) == 45
assert set(g_train).isdisjoint(g_test)
assert len(predictions) == len(X_test) and predictions.notna().all().all()
print('Verified: five trees, four final SVMs, 45 recorded outer-fold evaluations.')
print('Saved comparison tables, fold scores, predictions, partitions and metadata.')
''')
md('''## Findings and recommendations
{{FINDINGS}}

## Reflection and practical limitations
Engineered features combine physical quantities and can reduce the burden on a
shallow tree, but redundant information does not guarantee improvement. K-Best
retains named features and is easier to explain than PCA. PCA may compress inputs
while discarding low-variance class information; explained variance is not accuracy.
SVM tuning is judged by measured differences, not by the expectation of improvement.

The group holdout asks a harder question than re-predicting recipes already seen
at another age. Exact recipe grouping cannot remove dependence from near-identical
mixtures or unrecorded experimental batches. Five folds on one small laboratory
dataset cannot establish reliability in field construction. The prior portfolio
explored this same dataset, so feature hypotheses are informed by earlier full-data
EDA even though fitted statistics are now kept within training folds. External
validation is still needed.

Quartile classes are approximately balanced in training, but test balance is not
forced. Discretisation loses MPa detail and arbitrary boundaries can turn similar
strengths into different classes. A future regression model should be evaluated
with MPa MAE/RMSE if numerical strength is the real decision need. For class use,
an ordinal model could explicitly penalise distant-band errors. Unknown curing
conditions, material origins and laboratory protocols limit transfer. A model
output is not certified material strength and must not replace qualified testing.

Next steps are repeated recipe-group splits, a wider but carefully validated
parameter search, a deduplication sensitivity study, and external measurements
with specimen, batch and curing identifiers. Reusing the present holdout to select
new features would make it validation data rather than an independent test.

## References
Yeh, I. (1998). Concrete Compressive Strength [Dataset]. UCI Machine Learning
Repository. https://doi.org/10.24432/C5PK67. CC BY 4.0. Original workbook and README
copied from Portfolio 1; source download recorded there as 6 October 2026.

Scikit-learn documentation, accessed 18 October 2026:
https://scikit-learn.org/stable/common_pitfalls.html ;
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html ;
https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html .

Course task brief and Portfolio Assessment 2 rubric supplied by the user; used as
reference. Portfolio 1 by Pham Quang Minh supplies the EDA motivation.
''')

NOTEBOOK = ROOT / 'Portfolio_Assessment_2_Pham_Quang_Minh.ipynb'
nb = nbformat.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'display_name': 'Python 3', 'name': 'python3', 'language': 'python'},
    'language_info': {'name': 'python', 'version': '3.12'}})
if '--export-only' not in sys.argv:
    nbformat.write(nb, NOTEBOOK)
    kd = ROOT / '.kernels' / 'portfolio2'
    kd.mkdir(parents=True, exist_ok=True)
    env = {key: os.environ[key] for key in ['MPLCONFIGDIR', 'IPYTHONDIR']}
    if RUNTIME.exists():
        env['PYTHONPATH'] = str(RUNTIME)
    (kd / 'kernel.json').write_text(json.dumps({
        'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
        'display_name': 'Portfolio 2 build', 'language': 'python', 'env': env}))
    km = KernelManager(kernel_name='portfolio2',
        kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(ROOT / '.kernels')]))
    print('Executing Portfolio 2 notebook...', flush=True)
    NotebookClient(nb, km=km, timeout=1200).execute(cwd=str(ROOT))
    summary = json.loads((ROOT / 'results/summary.json').read_text())
    meta = json.loads((ROOT / 'results/reproducibility.json').read_text())
    delta = summary['tuning_CV_delta']
    findings = (
        f"The data contains {meta['clean_rows']} non-identical records after removing "
        f"{meta['duplicates_removed']} extra exact duplicates. The recipe split contains "
        f"{meta['train_rows']} training and {meta['test_rows']} holdout observations.\n\n"
        f"Among the five tree feature sets, **{summary['best_tree']}** achieved the "
        f"highest training CV macro F1 ({summary['tree_CV_F1']:.4f}); its holdout "
        f"macro F1 was {summary['tree_test_F1']:.4f}. Expanded-minus-original tree "
        f"CV F1 was {summary['expanded_tree_delta']:+.4f}, measuring the net effect "
        f"of the five engineered inputs.\n\n"
        f"Scaling changed baseline SVM CV macro F1 by {summary['scale_CV_delta']:+.4f}. "
        f"The tuned SVM changed outer CV macro F1 by {delta:+.4f} and holdout "
        f"macro F1 by {summary['tuning_test_delta']:+.4f} versus the baseline. "
        f"The direction and size of these changes are empirical results, not "
        f"a guarantee that tuning will improve a future dataset.\n\n"
        f"The CV-selected SVM is **{summary['best_svm']}**, with CV macro F1 "
        f"{summary['svm_CV_F1']:.4f}. On the holdout its accuracy is "
        f"{summary['svm_test_accuracy']:.4f}, macro F1 is {summary['svm_test_F1']:.4f}, "
        f"and mean absolute class error is {summary['svm_test_class_MAE']:.4f} bands. "
        f"The training-majority dummy has holdout accuracy {summary['dummy_accuracy']:.4f}. "
        f"This supports the selected SVM as a candidate for further validation, "
        f"not field deployment.\n\n"
        f"K-Best selected {meta['selected_K']} of 13 features. The PCA SVM retained "
        f"{meta['PCA_components']} components explaining {meta['PCA_variance']:.1%} "
        f"of training variance. Compression and predictive performance are distinct "
        f"criteria; compare their final-table F1 and interpretability before choosing.")
    for cell in nb.cells:
        if cell.cell_type == 'markdown':
            cell.source = cell.source.replace('{{FINDINGS}}', findings)
    assert all(c.execution_count is not None for c in nb.cells if c.cell_type == 'code')
    assert not any(o.output_type == 'error' for c in nb.cells if c.cell_type == 'code'
                   for o in c.outputs)
    nbformat.validate(nb)
    nbformat.write(nb, NOTEBOOK)
else:
    nb = nbformat.read(NOTEBOOK, as_version=4)

nb.cells[0].source = nb.cells[0].source.replace(
    'Student ID: __________', 'Student ID: 104999568')

# Keep the written explanations concise on every rebuild.
from plain_language import simplify_notebook
nb = simplify_notebook(nb, ROOT)
nbformat.write(nb, NOTEBOOK)
print('Notebook executed. Exporting report...', flush=True)

from reportlab.platypus import (SimpleDocTemplate, Paragraph, Preformatted, Image,
                               Spacer, PageBreak, Table, TableStyle, KeepTogether)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from xml.sax.saxutils import escape
import pandas as pd

font_dir = Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts'
if all((font_dir / n).exists() for n in ['arial.ttf', 'arialbd.ttf', 'ariali.ttf', 'cour.ttf']):
    for name, file in [('PortfolioArial', 'arial.ttf'),
                       ('PortfolioArial-Bold', 'arialbd.ttf'),
                       ('PortfolioArial-Italic', 'ariali.ttf'),
                       ('PortfolioMono', 'cour.ttf')]:
        pdfmetrics.registerFont(TTFont(name, str(font_dir / file)))
    pdfmetrics.registerFontFamily('PortfolioArial', normal='PortfolioArial',
        bold='PortfolioArial-Bold', italic='PortfolioArial-Italic',
        boldItalic='PortfolioArial-Bold')
    BODY_FONT, BOLD_FONT, MONO_FONT = 'PortfolioArial', 'PortfolioArial-Bold', 'PortfolioMono'
else:
    BODY_FONT, BOLD_FONT, MONO_FONT = 'Helvetica', 'Helvetica-Bold', 'Courier'

PAGE_WIDTH, PAGE_HEIGHT = 595.5, 842.25
CONTENT_WIDTH = PAGE_WIDTH - 108
styles = getSampleStyleSheet()
styles['Normal'].fontName = BODY_FONT
styles['Normal'].fontSize = 9.75
styles['Normal'].leading = 13.2
styles['Normal'].spaceAfter = 8
styles['Normal'].allowWidows = 0
styles['Normal'].allowOrphans = 0
for key in ['Heading1', 'Heading2']:
    styles[key].textColor = colors.HexColor('#184f70')
    styles[key].fontName = BOLD_FONT
    styles[key].keepWithNext = True
styles['Heading1'].fontSize = 18
styles['Heading1'].leading = 22
styles['Heading2'].fontSize = 12
styles['Heading2'].leading = 16
styles.add(ParagraphStyle('CodeLabel', fontName=BODY_FONT, fontSize=9, leading=12,
                          spaceAfter=5, spaceBefore=9, keepWithNext=True,
                          textColor='#184f70'))
styles.add(ParagraphStyle('CodeText', fontName=MONO_FONT, fontSize=7.5, leading=9.4,
                          spaceAfter=8))
styles.add(ParagraphStyle('Output', fontName=MONO_FONT, fontSize=7, leading=9,
                          spaceAfter=9))
styles.add(ParagraphStyle('TableCell', fontName=BODY_FONT, fontSize=8.5, leading=11))
styles.add(ParagraphStyle('References', fontName=BODY_FONT, fontSize=8.5,
                          leading=11, spaceAfter=6))
story = []
def inline(text):
    text = escape(text)
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    return re.sub(r'`([^`]+)`', rf'<font name="{MONO_FONT}">\1</font>', text)
def markdown(text):
    if '\n## References\n' in text:
        body, references = text.split('\n## References\n', 1)
        markdown(body)
        block = [Paragraph('References', styles['Heading2'])]
        for item in re.split(r'\n\s*\n', references.strip()):
            block.append(Paragraph(inline(' '.join(item.splitlines())),
                                   styles['References']))
        story.append(KeepTogether(block))
        return
    paragraph = []
    def flush():
        if paragraph:
            story.append(Paragraph(inline(' '.join(paragraph)), styles['Normal']))
            paragraph.clear()
    for line in text.splitlines():
        line = line.strip()
        if not line:
            flush()
        elif line.startswith('# '):
            flush()
            story.append(Paragraph(inline(line[2:]), styles['Heading1']))
        elif line.startswith('## '):
            flush()
            story.append(Paragraph(inline(line[3:]), styles['Heading2']))
        elif line.startswith('- '):
            flush()
            paragraph.append('- ' + line[2:])
        else:
            paragraph.append(line)
    flush()
def pre(text, style, width):
    lines = []
    for line in text.splitlines():
        lines.extend(textwrap.wrap(line.expandtabs(4), width=width,
                     replace_whitespace=False, drop_whitespace=False) or [''])
    # Keep a cell label with a short start, not an entire multi-page code block.
    if len(lines) > 6:
        start_style = ParagraphStyle(style.name + 'Start', parent=style, spaceAfter=0)
        story.append(Preformatted('\n'.join(lines[:6]), start_style))
        story.append(Preformatted('\n'.join(lines[6:]), style))
    else:
        story.append(Preformatted('\n'.join(lines), style))
def export_cell(cell):
    if cell.cell_type == 'markdown':
        markdown(cell.source)
    else:
        story.append(Paragraph(f'Code cell [{cell.execution_count}]', styles['CodeLabel']))
        pre(cell.source, styles['CodeText'], 104)
        for output in cell.outputs:
            if output.output_type == 'stream':
                pre(output.text, styles['Output'], 115)
            elif 'image/png' in output.get('data', {}):
                buf = io.BytesIO(base64.b64decode(output.data['image/png']))
                iw, ih = ImageReader(buf).getSize()
                ratio = min(CONTENT_WIDTH / iw, 520 / ih)
                story.append(Image(buf, width=iw * ratio, height=ih * ratio))
                story.append(Spacer(1, 9))
            elif 'text/plain' in output.get('data', {}):
                pre(output.data['text/plain'], styles['Output'], 115)

def report_table(path, selected_columns, labels):
    frame = pd.read_csv(ROOT / path)
    values = [labels]
    for _, row in frame.iterrows():
        values.append([str(row[c]) if isinstance(row[c], str) else
                       str(int(row[c])) if c == 'features' else f'{row[c]:.3f}'
                       for c in selected_columns])
    values = [[Paragraph(escape(v), styles['TableCell']) for v in row] for row in values]
    widths = [119] + [(CONTENT_WIDTH - 119) / (len(labels) - 1)] * (len(labels) - 1)
    table = Table(values, colWidths=widths, repeatRows=1, hAlign='LEFT')
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dceaf1')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f3f7f9')]),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
        ('LINEBELOW', (0, 0), (-1, 0), .7, colors.HexColor('#184e6a'))]))
    story.append(table)
    story.append(Spacer(1, 12))

# Portfolio 1 uses a dedicated centred cover, followed by declaration and report.
story.append(Spacer(1, 700))
story.append(PageBreak())
declaration_and_scope = nb.cells[0].source.split('## Generative AI declaration', 1)[1]
markdown('## Generative AI declaration' + declaration_and_scope)
markdown('# Report summary')
markdown('After removing duplicates, the dataset has 1,005 records: 676 for training '
         'and 329 for testing. The main results are below. Full code and outputs follow.')
final_markdown = nb.cells[-1].source
findings_only = final_markdown.split('## Limits and next steps')[0]
markdown(findings_only)
markdown('## First comparison: five decision-tree feature sets')
report_table('results/first_decision_tree_comparison.csv',
    ['model', 'CV_macro_F1', 'test_accuracy', 'test_macro_f1'],
    ['Decision tree', 'CV macro F1', 'Test accuracy', 'Test macro F1'])
markdown('## Final comparison: four SVM variations')
report_table('results/final_svm_comparison.csv',
    ['model', 'CV_macro_F1', 'test_accuracy', 'test_macro_f1'],
    ['SVM variation', 'CV macro F1', 'Test accuracy', 'Test macro F1'])
markdown('CV checks performance across five training groups. Macro F1 gives each '
         'strength class equal weight. Accuracy is the proportion predicted correctly. '
         'Higher scores are better. Test results use recipes kept out of training.')
markdown('# Complete executed notebook')
for cell in nb.cells[1:-1]:
    export_cell(cell)
markdown('## Limits and next steps' + final_markdown.split('## Limits and next steps', 1)[1])

PDF = ROOT / 'Portfolio_Assessment_2__Pham_Quang_Minh.pdf'
def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont(BODY_FONT, 8.25)
    canvas.setFillColor(colors.red)
    canvas.drawString(48.05, 24.775, 'Pham Quang Minh | 104999568 | COS40007 Portfolio 2')
    canvas.drawRightString(PAGE_WIDTH - 48.05, 24.775, f'Page {doc.page}')
    canvas.restoreState()

def cover(canvas, doc):
    footer(canvas, doc)
    canvas.saveState()
    canvas.setFillColor(colors.HexColor('#184f70'))
    canvas.setFont(BOLD_FONT, 18)
    canvas.drawCentredString(PAGE_WIDTH / 2, 603,
                            'Advanced Data Preprocessing,')
    canvas.drawCentredString(PAGE_WIDTH / 2, 579,
                            'Feature Engineering & Model Development')
    canvas.setFillColor(colors.black)
    canvas.setFont(BOLD_FONT, 9.75)
    canvas.drawCentredString(PAGE_WIDTH / 2, 443.65,
                            'Pham Quang Minh | Student ID: 104999568')
    canvas.setFont(BODY_FONT, 9.75)
    canvas.drawCentredString(PAGE_WIDTH / 2, 421.88,
        'COS40007 Artificial Intelligence Engineering - Portfolio Assessment 2')
    canvas.drawCentredString(PAGE_WIDTH / 2, 399.35,
        'Date: 18 October 2026 | Swinburne University of Technology')
    canvas.restoreState()

SimpleDocTemplate(str(PDF), pagesize=(PAGE_WIDTH, PAGE_HEIGHT), leftMargin=54,
    rightMargin=54, topMargin=24, bottomMargin=45,
    title='Portfolio Assessment 2 - Pham Quang Minh', author='Pham Quang Minh').build(
        story, onFirstPage=cover, onLaterPages=footer)

bundle_files = [NOTEBOOK, PDF, ROOT/'README.md', ROOT/'requirements.txt', ROOT/'.gitignore']
for folder in ['data', 'figures', 'results']:
    bundle_files += [p for p in (ROOT/folder).iterdir() if p.is_file()]
with zipfile.ZipFile(ROOT/'portfolio-2-submission.zip', 'w', zipfile.ZIP_DEFLATED) as z:
    for path in bundle_files:
        if path.exists():
            z.write(path, Path('portfolio-2') / path.relative_to(ROOT))
print('Created notebook, PDF and submission ZIP.', flush=True)
