'''
Created on Apr 14, 2021

@author: immanueltrummer
'''
import re

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