'''
Created on Apr 10, 2021

@author: immanueltrummer

Common utility functions used by multiple baselines.
'''
import re
import pandas as pd

def get_values(sentence):
    """ Returns candidate parameter values from given sentence. """
    candidates = []
    s_cleaned = re.sub(',|;|:|"|“|\u201F|\u201D|\u201C|\u0022\)|\(', 
                       ' ', sentence)
    for s in s_cleaned.split():
        if re.match(r'\d+.*|off$', s):
            candidates.append(s)
    return candidates

def get_parameters(sentence):
    """ Returns candidate parameters from given sentence. """
    s_cleaned = re.sub(r',|;|:|"|“|”|\)|\(', ' ', sentence)
    candidates = []
    for s in s_cleaned.split():
        if "_" in s:
            candidates.append(s)
    return candidates

def clean_sentence(sentence):
    """ Separate lower case letters, followed by upper case letters. """
    return re.sub(r'([a-z])([A-Z])', r'\1 \2', sentence)

def read_hints(docs_path):
    """ Returns list of sentences that may contain tuning hints. """
    print(f'Reading hints from {docs_path} ...')
    data = pd.read_csv(docs_path)
    data.fillna('', inplace=True)
    data['sentence'] = data.apply(
        lambda row: clean_sentence(row['sentence']), axis = 1)
    return data.groupby('filenr')['sentence'].apply(list)

def print_assignments(asg_to_doc):
    """ Prints out information on assignments with associated source documents. """
    print('\nExtracted assignments with corresponding source document IDs:')
    print('\n'.join(f'{k}, {v}' for k, v in asg_to_doc.items()))
    print('The following assignments are proposed by at least two sources:')
    verified_asgs = [k for k, v in asg_to_doc.items() if len(v)>=2]
    print('\n'.join(f'{k}, {v}' for k, v in verified_asgs))

def aggregate_asg_hints(assignments):
    """ Aggregate hints on value assignments from different sources. """
    pass