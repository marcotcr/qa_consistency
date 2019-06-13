# Evaluating consistency of Question-Answering Models
This repository contains code for creating implications and evaluating the consistency of question-answering models, as described in the following paper:
>[Are Red Roses Red? Evaluating Consistency of Question-Answering Models](http://homes.cs.washington.edu/~marcotcr/implication_acl19.pdf)  
> Marco Tulio Ribeiro, Sameer Singh, Carlos Guestrin  
> Association for Computational Linguistics (ACL), 2019



## Installation
1. Create and activate a virtual environment, e.g.:
```
virtualenv -p python3.6 env
source env/bin/activate
```
2. Clone this repository and cd to the folder:
```
git clone git@github.com:marcotcr/qa_consistency.git
cd qa_consistency
```
3. Run the following :
```
pip install .
git clone https://github.com/kelvinguu/qanli.git
python -c "import benepar;benepar.download('benepar_en_small')"
python -m spacy download en_core_web_sm
```

## Generating implications:
### VQA
```python
import qa_consistency
import qa_consistency.implication
gen = qa_consistency.implication.ImplicationsVQA()
gen.implications('How many birds?', '3')
```
> [('Are there 3 birds ?', 'yes', 'yeseqcount'),  
 ('Are there 4 birds ?', 'no', 'n+1'),  
 ('Are there any birds ?', 'yes', 'ans>0 implies some')]

 ### SQuAD

```python
import qa_consistency
import qa_consistency.implication
gen = qa_consistency.implication.ImplicationsSquad()
passage = 'Kublai originally named his eldest son, Zhenjin, as the Crown Prince, \
but he died before Kublai in 1285.'
gen.implications('When did Zhenjin die?', '1285', passage)
```
> [('Who died in 1285?', 'Zhenjin', 'subj')]

## Evaluating the consistency of models
### VQA
Download and extract precomputed implications [here](https://github.com/marcotcr/qa_consistency/blob/master/precomputed/vqa_imps.zip).
Create a folder for the consistency dataset (`CONSISTENCY_FOLDER`). Output your model predictions into a json file (`PRED_FILE`) in the VQA format. Then run:
```python
import qa_consistency
import qa_consistency.dataset_utils
all_imps = pickle.load(open('vqa_imps.pkl', 'rb'))
vqa = qa_consistency.dataset_utils.load_vqa(vqa_path, 'validation')
# Uncomment the line below if you want vqa v2
# vqa = qa_consistency.dataset_utils.load_vqav2(vqa_path, 'validation')
qa_consistency.dataset_utils.generate_implication_vqa(vqa, PRED_FILE, all_imps, CONSISTENCY_FOLDER)
```
This will write `CONSISTENCY_FOLDER/{questions,annotations}.json`.
At this point you should run your model on these files, and generate a new prediction file (CONSISTENCY_PRED_FILE), and then run:

```python
stats = qa_consistency.dataset_utils.evaluate_consistency_vqa(CONSISTENCY_FOLDER, CONSISTENCY_PREDS_FILE)
print('Consistency by implication type:')
print()
for x, v in stats.items():
    if x == 'all':
        continue
    print('%s : %.1f' % (x, 100* v))
print()
print('Avg  : %.1f' % (100 * stats['all']))
```

### SQuAD
Download and extract precomputed implications [here](https://github.com/marcotcr/qa_consistency/blob/master/precomputed/squad_imps.zip).
Let `SQUAD_PATH` be a pointer to the original squad dev set json (dev-v1.1.json), `PRED_FILE` be the predictions json on the dev set from your model in the SQuAD official format (dictionary of id : answer). Run:
```python
import qa_consistency
import qa_consistency.dataset_utils
all_imps = pickle.load(open('squad_imps.pkl', 'rb'))
qa_consistency.dataset_utils.generate_implication_squad(
SQUAD_PATH, PRED_FILE, all_imps, NEW_SQUAD_JSON)
```
This will generate a new dataset in the SQuAD format in the `NEW_SQUAD_JSON` path. At this point you should run your model on this file, and generate a new prediction file (`CONSISTENCY_PRED_FILE`), and then run:
```python
stats = qa_consistency.dataset_utils.evaluate_consistency_squad(NEW_SQUAD_JSON, CONSISTENCY_PRED_FILE)
print('Consistency by implication type:')
print()
for x, v in stats.items():
    if x == 'all':
        continue
    print('%s : %.1f' % (x, 100* v))
print()
print('Avg  : %.1f' % (100 * stats['all']))
```

## Notebooks where we bring it all together
- [VQA notebook](https://github.com/marcotcr/qa_consistency/blob/master/notebooks/VQA.ipynb)
- [SQuAD notebook](https://github.com/marcotcr/qa_consistency/blob/master/notebooks/SQuAD.ipynb)

### Code of Conduct
[Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct)
