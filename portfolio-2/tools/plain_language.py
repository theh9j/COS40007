"""Keep the notebook explanations short; leave analysis code and outputs intact."""
import json
import textwrap


def simplify_notebook(nb, root):
    result = json.loads((root / 'results/summary.json').read_text())
    meta = json.loads((root / 'results/reproducibility.json').read_text())
    texts = {
        0: '''
        # Advanced Data Preprocessing, Feature Engineering & Model Development
        **Pham Quang Minh | Student ID: 104999568**

        COS40007 Artificial Intelligence Engineering | Portfolio Assessment 2

        Date: 18 October 2026 | Studio number: __________

        ## Generative AI declaration
        OpenAI Codex helped draft the code, explanations and figures, and run checks.
        I will review and verify the work before submission.

        ## Aim
        Use concrete ingredients and age to predict one of four strength groups.
        Compare decision trees and support vector machines (SVMs).
        The dataset is included. Run the notebook cells in order from this folder.
        ''',
        1: '''
        # 1. Prepare the target
        Portfolio 1 found no missing values and 25 extra duplicate rows. After removing
        these duplicates, 1,005 records remain.

        Strength is measured in MPa. Split it into four groups: Lower, Lower-middle,
        Upper-middle and Higher. Training-data quartiles make the groups roughly equal
        in size. These labels describe this dataset; they are not construction grades.
        ''',
        3: '''
        ## Training and test data
        Keep tests of the same concrete recipe together, even when their ages differ.
        This prevents the model seeing the same recipe in training and testing.
        A 70/30 split of recipes gives 676 training rows and 329 test rows.

        Keep valid zero ingredient amounts and unusual values. No missing-value
        replacement is needed.
        ''',
        5: '''
        ## How models are checked
        Five-fold cross-validation (CV) trains on four parts and checks the fifth,
        repeating until each part has been checked. The same recipe stays in one part.
        Class boundaries and data transformations are learned only from training data.
        When tuning a model, another five-fold check chooses settings inside each
        training part. The separate test set is not used to choose settings.

        Accuracy is the proportion of correct predictions. Macro F1 balances precision
        and recall while giving each class equal weight. Higher is better. Use CV
        macro F1 to choose the model. Class MAE shows the average number of bands missed.
        ''',
        6: '''
        # 2. Scale the inputs
        Scaling puts measurements on comparable scales:

        - Use StandardScaler for water, coarse aggregate and fine aggregate, which
          were relatively symmetric in Portfolio 1.
        - Use RobustScaler for cement, slag, fly ash, superplasticizer and age, which
          were more uneven or had extreme values.

        All inputs are numeric, so no category encoding is needed. The four target
        groups are represented by codes 0-3.
        ''',
        8: '''
        Scaling helps SVMs compare inputs fairly, so large ingredient quantities do not
        overwhelm smaller measurements. It usually matters less for decision trees.
        Before PCA, inputs are also standardised to equal variance.
        ''',
        9: '''
        # 3. Create useful features
        Portfolio 1 suggested looking at ingredient ratios and the effect of age.
        Add five features:

        - `binder_total = cement + slag + fly_ash`: total binder amount (kg/m^3).
        - `water_binder_ratio = water / binder_total`: water relative to binder.
        - `log_age = log(1 + age)`: reduces the effect of very large ages.
        - `scm_fraction = (slag + fly_ash) / binder_total`: share of replacement binder.
        - `maturity_ratio = log_age / water_binder_ratio`: an experimental age-water
          combination, not a standard engineering maturity measure.

        The checks below confirm the values are valid and show their relationships
        with training strength. No feature uses the measured target strength.
        ''',
        11: '''
        Some new features overlap with the original inputs. Compare model results to
        see whether the extra information helps.
        ''',
        12: '''
        # 4. Compare five decision trees
        Test five input sets:

        - Original: all eight original inputs.
        - Top 5: five inputs most correlated with training strength.
        - Expanded: eight original inputs plus five new features.
        - PCA 5: five combined inputs produced by PCA.
        - Domain: five inputs chosen for their engineering meaning.

        Every tree uses maximum depth 6 and at least 5 records per leaf. Using the
        same settings makes the feature-set comparison easier to interpret.
        ''',
        15: '''
        # 5. Decision-tree results
        Compare all five trees using the same split and CV groups. Select the best
        tree by CV macro F1, then check its test performance.
        ''',
        17: f'''
        **DT Domain** has the best tree CV macro F1 ({result['tree_CV_F1']:.3f}).
        Water-to-binder ratio and log age are its most important inputs. This suggests
        a small set of meaningful engineering features can work well.
        The importance chart describes this fitted tree, not cause and effect.
        ''',
        18: '''
        # 6. Try an SVM
        An SVM learns boundaries between the four strength groups. Start with all
        13 scaled inputs, an RBF kernel, C=1 and gamma='scale'. Also check an unscaled
        SVM to see how much scaling helps.
        ''',
        20: '''
        # 7. Tune the SVM
        GridSearchCV tries nine combinations of C and gamma. C controls the penalty
        for training errors; gamma controls how locally each point affects the
        boundary. Choose settings using training CV, not the test set.
        ''',
        22: f'''
        # 8. Select fewer features
        K-Best ranks features by how well they separate the classes. Try keeping
        5, 8 or 11 inputs, and let training CV choose the setting.
        The final model keeps **{meta['selected_K']} of 13 features**. Their names appear below.
        ''',
        24: f'''
        # 9. Reduce inputs with PCA
        PCA combines related inputs into fewer components. Try retaining 80%, 90%
        or 95% of input variance and choose using training CV.
        The selected model uses **{meta['PCA_components']} components**, keeping about
        **{meta['PCA_variance']:.0%}** of training variance. Keeping variance does not guarantee
        better predictions.
        ''',
        26: '''
        # 10. Final comparison
        Compare the baseline, tuned, K-Best and PCA SVMs. Choose using CV macro F1;
        use the separate test results as a final check. The baseline is fixed, while
        CV for the other models includes their tuning steps.
        ''',
        30: f'''
        ## Main findings
        - The best decision-tree input set is **DT Domain**, with CV macro F1
          **{result['tree_CV_F1']:.3f}**.
        - Choose **{result['best_svm']}** because it has the highest SVM CV macro F1
          (**{result['svm_CV_F1']:.3f}**). It correctly predicts **{result['svm_test_accuracy']:.1%}** of test records.
        - PCA scores slightly higher on this one test set, but it was not the CV
          winner. Keep the original CV-based choice instead of choosing by test results.

        ## Limits and next steps
        These groups describe relative strength, not exact MPa values or construction
        grades. The dataset is small, related recipes may remain, and Portfolio 1
        already informed the feature ideas. More outside data is needed before
        practical use. The tuning grids also differ, so model differences cannot be
        explained only by feature selection or PCA.

        Next, check more recipe splits and compare a regression model that predicts
        strength directly in MPa.

        ## References
        Yeh, I. (1998). Concrete Compressive Strength [Dataset]. UCI Machine Learning
        Repository. https://doi.org/10.24432/C5PK67. CC BY 4.0.

        Scikit-learn: pipelines, grouped splitting and nested cross-validation.
        https://scikit-learn.org/stable/common_pitfalls.html ;
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html ;
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html .

        Course task and rubric; Portfolio 1 by Pham Quang Minh.
        '''}
    for index, text in texts.items():
        assert nb.cells[index].cell_type == 'markdown'
        nb.cells[index].source = textwrap.dedent(text).strip()
    return nb
