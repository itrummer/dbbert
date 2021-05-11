'''
Created on Apr 10, 2021

@author: immanueltrummer

Reimplements supervised learning baseline from VLDB'21 vision paper.
'''
import argparse
import pandas as pd
import random as rand
import torch.cuda
from baselines.common import clean_sentence
from simpletransformers.classification import ClassificationModel, ClassificationArgs

def label_formula_ops(row):
    """ Label formula based on operators that appear in it. """
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

def has_param(sentences, i):
    """ True iff i-th sentence is likely to contain a parameter name. """
    return ("_" in sentences['sentence'].iloc[i])    

# For reproducible results
rand.seed(42)

def get_context(cur_sentences, i):
    """ Returns most likely context for i-th sentence. """
    for j in range(i, -1, -1):
        if has_param(cur_sentences, j):
            return j
    return 0

def label_hints(hints):
    """ Label hints for training or testing. """
    hints['sentence'] = hints.apply(
        lambda row: clean_sentence(row), axis = 1)
    hints['ops_label'] = hints.apply(
        lambda row: label_formula_ops(row), axis = 1)
    hints['key_label'] = hints.apply(
        lambda row: 1 if row['KeySentence'] == 1 else 0, axis = 1)

if __name__ == '__main__':
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Trains models for extracting tuning hints')
    parser.add_argument('train_path', type=str, help='Path to .csv file with labeled hints')
    parser.add_argument('out_detect', type=str, help='Path to detect model output')
    parser.add_argument('out_classify', type=str, help='Path to classification model output')
    args = parser.parse_args()
    
    # Read and pre-process training data
    training = pd.read_csv(args.train_path, sep=",")
    training.fillna(value='', inplace=True)
    label_hints(training)
    
    train_detect = training[['sentence', 'key_label']]
    train_classify = training[['sentence', 'ops_label']]
    
    # Determine training device
    use_cuda = True if torch.cuda.is_available() else False
    
    # Train model for recognizing key sentences
    print('Training to detect key sentences with tuning hints ...')
    model_args = ClassificationArgs(num_train_epochs=10, train_batch_size=50,
                                    overwrite_output_dir=True)
    detect_model = ClassificationModel(model_type='roberta', use_cuda=use_cuda,
                                  model_name='roberta-base', args=model_args, 
                                  num_labels=2, weight=[1, 50])
    detect_model.args.no_save = True
    detect_model.train_model(train_detect)
    detect_model.save_pretrained(args.out_detect)
    
    # Train model for recognizing the type of tuning hint
    print('Training to classify tuning sentences ...')
    model_args = ClassificationArgs(num_train_epochs=20, train_batch_size=20,
                                    overwrite_output_dir=True)
    type_model = ClassificationModel(model_type='roberta', use_cuda=use_cuda,
                                model_name='roberta-base', args=model_args, 
                                num_labels=7)
    type_model.args.no_save = True
    type_model.train_model(train_classify)
    type_model.save_pretrained(args.out_classify)