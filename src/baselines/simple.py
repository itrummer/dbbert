'''
Created on Apr 10, 2021

@author: immanueltrummer

Re-implementation of the simple baseline used in the VLDB'21 vision paper.
'''
import argparse
from random import choice
from baselines.common import get_parameters, get_values, read_hints,\
    print_assignments
from collections import defaultdict

def detect_base(sentence):
    """ Detects key sentences using a simple heuristic. 
    
    Returns:
        True iff sentence seems to contain tuning hint
    """
    if get_parameters(sentence) and get_values(sentence):
        return True
    else:
        return False

def classify_base(sentence):
    """ Classify sentences based on a simple heuristic. """
    nr_vals = len(get_values(sentence))
    if nr_vals==1:
        return choice([0, 1, 2, 3, 4])
    elif nr_vals>1:
        return 5
    else:
        return 6
    
# Parse command line arguments
parser = argparse.ArgumentParser(description='Extracts tuning hints from given documents')
parser.add_argument('doc_path', type=str, help='Specify path to .csv file with tuning hints')
args = parser.parse_args()

# Collect assignment hints from input documents
asg_to_doc = defaultdict(lambda: set())
doc_sentences = read_hints(args.doc_path)
for doc_id, sentences in doc_sentences.iteritems():
    for sentence in sentences:
        if detect_base(sentence) and classify_base(sentence) == 4:
            # Likely value assignment hint - extract all combinations
            for param in get_parameters(sentence):
                for value in get_values(sentence):
                    assignment = (param, value)
                    asg_to_doc[assignment].add(doc_id)

# Print out information on assignments
print_assignments(asg_to_doc)