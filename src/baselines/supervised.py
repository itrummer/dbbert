'''
Created on Apr 10, 2021

@author: immanueltrummer

Reimplements supervised learning baseline from VLDB'21 vision paper.
'''
import pandas as pd
import random as rand
import re
from sklearn.metrics import matthews_corrcoef as mcc
from sklearn.metrics import confusion_matrix as cm

# Configure DBMS to use for training and testing
train_dbms = ['pg']
test_dbms = ['ms']

# Load and label data
path = '/content/drive/My Drive/Colab Notebooks/Liter/webtext/AllSentences2.csv'
sentences = pd.read_csv(path, sep=",")
sentences.fillna(value='', inplace=True)

# For reproducible results
rand.seed(42)

def label_formula_ops(row):
  # Label based on formula operator
  str_form = str(row['Formula'])
  if "<" in str_form and ">" in str_form:
    return 0 # Range
  if "<" in str_form:
    return 1 # LB
  elif ">" in str_form:
    return 2 # UB
  elif "!=" in str_form:
    return 3 # NE
  elif "=" in str_form:
    return 4 # EQ
  elif "in" in str_form:
    return 5 # Set
  else:
    return 6 # "NA"

def clean_sentence(row):
  # Separate lowercase letter, followed by upper case
  return re.sub(r'([a-z])([A-Z])', r'\1 \2', str(row['sentence']))

# Whether i-th sentence contains likely parameter name
def has_param(sentences, i):
  return ("_" in sentences['sentence'].iloc[i])

# Analyzing context in all sentences
nr_sentences = sentences.shape[0]
other_p_ctr = 0
for i in range(0, nr_sentences):
  if sentences['KeySentence'].iloc[i] == 1:
    context = str(sentences['Context'].iloc[i])
    if len(context)>0:
      print(sentences['Context'].iloc[i])
      int_ctx = int(float(context))-2
      ctx_sentence = sentences['sentence'].iloc[int_ctx]
      print(ctx_sentence)
      print(has_param(sentences, int_ctx))
      other_params = False
      for j in range(int_ctx+1,i):
        if has_param(sentences, j):
          other_params = True
          other_p_ctr += 2
      print(f'Other params: {other_params}')
print(f'Nr. other params in between: {other_p_ctr}')

# Returns most likely context for i-th sentence
def get_context(cur_sentences, i):
  nr_cur_s = cur_sentences.shape[0]
  for j in range(i, -1, -1):
    if has_param(cur_sentences, j):
      return j
  return 0



# Label sentences according to different criteria
sentences['sentence'] = sentences.apply(
    lambda row: clean_sentence(row), axis = 1)
sentences['ops_label'] = sentences.apply(
    lambda row: label_formula_ops(row), axis = 1)
sentences['key_label'] = sentences.apply(
    lambda row: 1 if row['KeySentence'] == 1 else 0, axis = 1)

# Separate data by underlying system
pg_data = sentences[sentences['dbms']=='pg']
ms_data = sentences[sentences['dbms']=='ms']

# Collect training data for recognizing tuning hint types
key_sentences = sentences[sentences['KeySentence']==1]
key_pg = key_sentences[key_sentences['dbms']=='pg']
key_ms = key_sentences[key_sentences['dbms']=='ms']

# Prepare test and training data for detecting tuning hints
dbms_to_all = {'pg' : pg_data[['sentence', 'key_label']], 
               'ms' : ms_data[['sentence', 'key_label']]}
train_detect = pd.concat([dbms_to_all[d] for d in train_dbms])
test_detect = pd.concat([dbms_to_all[d] for d in test_dbms])
pd.set_option("display.max_rows", None, "display.max_columns", None)
print(train_detect.info())
print(test_detect.info())

# Collect test and training data for classifying hints
dbms_to_class = {'pg' : key_pg[['sentence', 'ops_label']], 
                'ms' : key_ms[['sentence', 'ops_label']]}
train_class = pd.concat([dbms_to_class[d] for d in train_dbms])
test_class = pd.concat([dbms_to_class[d] for d in test_dbms])
print(train_class.head())
print(test_class.head())



# Evaluate simple baseline for detecting key sentences
print(f'Training: {train_dbms}; Testing: {test_dbms}')
print('Performance of baseline for detecting key sentences')
pred_det = [detect_base(s) for s in test_detect['sentence']]
print(mcc(test_detect['key_label'], pred_det))
print(cm(test_detect['key_label'], pred_det))

# Evaluate simple baseline for classifying key sentences
pred_class = [classify_base(s) for s in test_class['sentence']]
print('Performance of baseline for classifying key sentences')
print(mcc(test_class['ops_label'], pred_class))
print(cm(test_class['ops_label'], pred_class, labels=list(range(0,7))))