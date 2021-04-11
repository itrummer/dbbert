'''
Created on Apr 11, 2021

@author: immanueltrummer
'''
import argparse
from baselines.common import read_hints, get_parameters, get_values, print_assignments
from collections import defaultdict
from simpletransformers.classification import ClassificationModel

def has_param(sentence):
    """ Determine whether sentence contains parameter via simple heuristic. """
    return ("_" in sentence)

def get_context(cur_sentences, i):
    """ Returns most likely context for i-th sentence. """
    for j in range(i, -1, -1):
        if has_param(cur_sentences[j]):
            return j
    return 0

if __name__ == '__main__':
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extracts tuning hints via supervised learning')
    parser.add_argument('detect_model_dir', type=str, help='Directory containing detection model')
    parser.add_argument('classify_model_dir', type=str, help='Directory containing classification model')
    parser.add_argument('docs_path', type=str, help='Path of file containing tuning hints')
    args = parser.parse_args()

    # Load models from given directories
    detect_model = ClassificationModel('roberta', args.detect_model_dir, use_cuda=False)
    print('Detection model loaded.')    
    classification_model = ClassificationModel('roberta', args.classify_model_dir, use_cuda=False)
    print('Classification model loaded.')
    
    # Collect assignment hints from input documents
    asg_to_doc = defaultdict(lambda: set())
    doc_sentences = read_hints(args.docs_path)
    for doc_id, sentences in doc_sentences.iteritems():
        nr_sentences = len(sentences)
        key_prediction = detect_model.predict(sentences)
        type_prediction = classification_model.predict(sentences)        
        for i in range(0, nr_sentences):
            if key_prediction[0][i] == 1:
                pred_type = type_prediction[0][i]
                sentence = sentences[i]
                values = get_values(sentence)
                ctx_idx = get_context(sentences, i)
                ctx_sent = sentences[ctx_idx]
                params = set(get_parameters(ctx_sent))
                for param in params:
                    for value in values:
                        assignment = (param, value)
                        asg_to_doc[assignment].add(doc_id)
    
    # Print out extracted assignments
    print_assignments(asg_to_doc)