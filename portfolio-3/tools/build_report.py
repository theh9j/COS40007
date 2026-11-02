"""Build the concise report from the saved experiment results; no model training."""
from pathlib import Path
import csv
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'Portfolio_Assessment_3_104999568_Pham_Quang_Minh.pdf'
for name, file in [('Arial','arial.ttf'),('Arial-Bold','arialbd.ttf'),('Arial-Italic','ariali.ttf')]:
    pdfmetrics.registerFont(TTFont(name, 'C:/Windows/Fonts/' + file))
pdfmetrics.registerFontFamily('Arial', normal='Arial', bold='Arial-Bold', italic='Arial-Italic')
BLUE = colors.HexColor('#184f70')
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='Body',fontName='Arial',fontSize=9.75,leading=13.2,spaceAfter=8))
styles.add(ParagraphStyle(name='Title3',fontName='Arial-Bold',fontSize=18,leading=22,textColor=BLUE,spaceAfter=14,keepWithNext=True))
styles.add(ParagraphStyle(name='Sub3',fontName='Arial-Bold',fontSize=12,leading=16,textColor=BLUE,spaceAfter=8,keepWithNext=True))
styles.add(ParagraphStyle(name='Small3',parent=styles['Body'],fontSize=8.5,leading=11))
story=[]
def p(text, style='Body'): story.append(Paragraph(text,styles[style]))
def title(text): p(text,'Title3')
def sub(text): p(text,'Sub3')
def fig(file,caption,width=487):
    from PIL import Image as PILImage
    with PILImage.open(ROOT/'figures'/file) as im: w,h=im.size
    story.append(Image(str(ROOT/'figures'/file),width=width,height=width*h/w))
    p(caption,'Small3')
def table(rows,widths):
    rows=[[Paragraph(str(x),styles['Small3']) for x in row] for row in rows]
    t=Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eaf0f4')),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.6,BLUE),('LINEBELOW',(0,1),(-1,-1),.25,colors.lightgrey),('LEFTPADDING',(0,0),(-1,-1),6),('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]))
    story.extend([t,Spacer(1,10)])
def page(): story.append(PageBreak())
def footer(c,doc):
    c.saveState(); c.setFillColor(colors.red); c.setFont('Arial',8.25)
    c.drawString(48.05,24.775,'Pham Quang Minh | 104999568 | COS40007 Portfolio 3')
    c.drawRightString(595.5-48.05,24.775,str(doc.page)); c.restoreState()
def cover(c,doc):
    footer(c,doc); c.saveState(); c.setFillColor(BLUE); c.setFont('Arial-Bold',18)
    c.drawCentredString(297.75,603,'Portfolio Assessment 3')
    c.drawCentredString(297.75,579,'Advanced ML and Deep Learning')
    c.setFillColor(colors.black); c.setFont('Arial-Bold',9.75)
    c.drawCentredString(297.75,443.65,'Pham Quang Minh | Student ID: 104999568')
    c.setFont('Arial',9.75)
    c.drawCentredString(297.75,421.88,'COS40007 Artificial Intelligence Engineering - Portfolio Assessment 3')
    c.drawCentredString(297.75,399.35,'Date: 2 November 2026 | Swinburne University of Technology')
    c.drawCentredString(297.75,364,'Option 3d: Classification with a tabular ANN')
    c.drawCentredString(297.75,342,'GitHub: https://github.com/theh9j/COS40007')
    c.restoreState()

story.append(Spacer(1,1)); page()
title('1. Problem and data preparation')
p('The aim is to predict a concrete sample\'s relative strength class using its ingredients and age. The four classes are Lower, Lower-middle, Upper-middle and Higher. They continue Portfolio 2 and are <b>relative dataset groups, not engineering safety grades</b>.')
sub('GenAI declaration')
p('OpenAI Codex assisted with model implementation, adaptation of the supplied teaching examples, checks, explanations and report formatting. The personal reflection is my own account, lightly edited for grammar with AI assistance.')
sub('What was kept consistent')
table([['Item','Preparation'],['Data','1,030 source records; 25 duplicate copies removed in earlier preparation; 1,005 retained. No missing values.'],['Inputs','Cement, slag, fly ash, water, superplasticizer, coarse aggregate, fine aggregate and age. Strength, labels and IDs excluded.'],['Target','Fixed Portfolio 2 cutoffs: 23.2393, 33.4542 and 44.3436 MPa (rounded here). The notebook uses the full saved values.'],['Outer split','676 training rows and 329 test rows. Samples with the same ingredient recipe stay together, including samples measured at different ages.'],['Scaling','StandardScaler learns only from fitting rows, including separately inside each traditional CV fold.']],[85,402])
p('Keeping related recipes together reduces leakage: otherwise a model could see almost the same mixture during training and testing. The saved outer split is reused for every model. Earlier portfolios already explored this dataset, so this test set is a comparison set rather than independent external validation.')
sub('How performance is measured')
p('Accuracy is the fraction of correct class predictions. Weighted F1 balances precision and recall while weighting each class by its number of records. Five-fold recipe-group cross-validation (CV) is used for the two traditional models. The ANN uses a separate validation set; its validation score is not a CV score.')
page()
title('2. Traditional model baselines')
p('Random Forest uses 200 trees and at least two records per leaf. The scikit-learn MLP uses hidden layers of 100 and 50 units, ReLU, Adam and a 500-iteration limit. Both use the same five recipe-group folds and scaler pipelines.')
table([['Model','Mean CV accuracy','Test accuracy','Test weighted F1'],['Random Forest','69.2%','78.1%','0.781'],['MLP','75.2%','82.1%','0.820']],[130,125,110,122])
fig('traditional_models_comparison.png','Figure 1. Traditional model comparison and MLP confusion matrix. CV error bars show fold standard deviations, not confidence intervals.')
sub('Interpretation')
p('The MLP is the traditional baseline selected by mean CV accuracy. Its test accuracy is also higher than the Random Forest\'s. Both models achieve about 97% training accuracy, well above CV performance, so their training scores are optimistic.')
p('<b>Limitation:</b> all six MLP fits (five folds and the final fit) reached 500 iterations without convergence. Its score is useful as a fixed baseline, but it should not be described as fully converged. Both complete scaler/model pipelines are saved as pickle files.')
page()
title('3. ANN design and first experiment')
p('A dense ANN suits these eight tabular inputs. The age column does not make the independent samples a time series, so an LSTM is unnecessary. Four softmax outputs represent the four classes.')
table([['Layer','Output units','Parameters'],['Dense + ReLU','64','576'],['Dropout (0.2)','64','0'],['Dense + ReLU','32','2,080'],['Dropout (0.2)','32','0'],['Dense + ReLU','16','528'],['Dense + softmax','4','68'],['Total trainable parameters','','3,252']],[245,120,122])
p('From the 676 training rows, 20% of recipe groups are reserved for validation: <b>513 fitting rows and 163 validation rows</b>. Group sizes vary, so the row percentage differs from 20%. The ANN scaler fits only those 513 rows.')
p('Settings: Adam learning rate 0.001, sparse categorical cross-entropy, batch size 32 and seed 42. Early stopping monitors validation loss with patience 10 and restores the best weights. The learning rate halves after five unimproved epochs. Saved checkpoints select the lowest validation loss.')
fig('ann_baseline_learning_curves.png','Figure 2. Training and validation loss/accuracy for the 10-epoch ANN.')
p('All 10 epochs completed, with the best validation loss at epoch 10. Test accuracy was <b>62.3%</b> and weighted F1 <b>0.583</b>. Loss was still improving, motivating the already planned longer run. Dropout is active during training but off during validation, so validation accuracy can exceed the training curve.')
page()
title('4. Controlled longer-training experiment')
p('Only the maximum epoch budget changed: <b>10 to 50</b>. The model restarted from the same seed; it did not continue from the first model\'s trained weights. Layers, scaler, recipe split, batch size and callbacks stayed fixed. The first ten epochs reproduce the original history.')
fig('ann_extended_learning_curves.png','Figure 3. Extended learning curves. The dashed line marks the original budget; the selected weights have the lowest validation loss.')
table([['Measure','10-epoch ANN','50-epoch ANN'],['Selected / completed epoch','10 / 10','50 / 50'],['Validation accuracy','60.1%','72.4%'],['Validation loss','0.908','0.652'],['Test accuracy','62.3%','75.7%'],['Test weighted F1','0.583','0.746'],['Test loss','0.932','0.602']],[227,130,130])
p('The longer run is the preferred ANN because its <b>validation loss is lower</b>, not because its test score was used for selection. Its test accuracy increased by 13.4 percentage points. All 50 epochs completed; early stopping did not activate before the limit. Reaching the limit does not prove convergence.')
p('Both ANN variants are saved in H5 format with native Keras checkpoints, the fitting-only scaler and pickled/CSV histories. Reload checks reproduced their predictions. No additional settings were tried after inspecting these test results.')
page()
title('5. Final comparison and conclusion')
rows=list(csv.DictReader((ROOT/'results/model_comparison.csv').open(encoding='utf-8')))
table([['Model','Test accuracy','Weighted F1','Mean CV accuracy']]+[[r['model'],f"{float(r['test_accuracy']):.1%}",f"{float(r['test_f1_weighted']):.3f}",f"{float(r['cv_accuracy_mean']):.1%}" if r['cv_accuracy_mean'] else 'Not run'] for r in rows],[165,100,100,122])
fig('four_model_comparison.png','Figure 4. Test accuracy for all four models and the extended ANN confusion matrix. All models use the same 329 test rows.')
p('<b>Main finding:</b> the MLP has the highest observed test accuracy (82.1%). More training improved the Keras ANN, but it did not exceed either traditional baseline in this experiment.')
p('The extended ANN recognises the extreme classes better: recall is 94.8% for Lower and 93.6% for Higher, compared with 54.2% and 57.0% for the two middle classes. The confusion matrix shows where its predictions need improvement.')
p('<b>Limits and next step:</b> the ANN fits 513 rows while traditional models fit 676, so this is not an isolated architecture comparison. Results come from one reused test set, one ANN seed and two planned budgets. Keep the MLP convergence warning visible. A future study could compare matched fitting sets, use grouped validation for tuning, repeat seeds, then assess a fresh external set. These class predictions are not a structural-safety assessment.')
page()
title('6. GenAI reflection and references')
sub('Tool and assistance record')
p('Tool: OpenAI Codex desktop coding agent. The precise underlying model/version was not captured in the project record and is not asserted here. Assistance covered adapting supplied model examples to concrete classes, implementing pipelines and the controlled ANN experiment, checking saved results, and preparing explanations and this PDF.')
p('The notebook contains the executed implementation and outputs. The following prompt excerpts document the iterative approach used with Codex:')
for q in ['Since we\'re not supposed to commit everything at once, lets try do things slowly but make sure is accurate','right, continue to the next iteration','alright, next iteration']:
    p('"'+q+'"','Small3')
p('The first quotation is an excerpt from the initial Assignment 3 request. The prompts requested staged progress; the assistance included the implementation and verification tasks declared above.')
sub('Personal learning reflection')
p('During this project, setting up and structuring the ANN model was the most tedious and difficult part because I had never attempted it before.')
sub('References and attribution')
p('Yeh, I. (1998). Concrete Compressive Strength. UCI Machine Learning Repository. DOI: <link href="https://doi.org/10.24432/C5PK67" color="#184f70">10.24432/C5PK67</link>. Data attribution retained from the earlier portfolios; CC BY 4.0. Prepared labels and recipe partitions continue Portfolio 2.','Small3')
p('Course materials: AssignmentTasks.pdf; Portfolio Task 3d - Advanced ML Models and Deep Learning - MLP for Classification.pdf; MLP_Classifcation.ipynb; ANN_Regression_Classification.ipynb; DL_Models_ExampleCode.ipynb. Teaching examples were adapted to the eight concrete inputs and four classes.','Small3')
p('Implementation references: scikit-learn RandomForestClassifier, MLPClassifier and grouped cross-validation APIs; Keras Sequential, Dense, Dropout and callback APIs.','Small3')
p('Project: <link href="https://github.com/theh9j/COS40007" color="#184f70">https://github.com/theh9j/COS40007</link>.','Small3')

SimpleDocTemplate(str(OUT),pagesize=(595.5,842.25),leftMargin=54,rightMargin=54,topMargin=24,bottomMargin=45,title='Portfolio Assessment 3 - Pham Quang Minh',author='Pham Quang Minh').build(story,onFirstPage=cover,onLaterPages=footer)
print(OUT)
