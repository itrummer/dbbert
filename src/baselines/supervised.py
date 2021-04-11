'''
Created on Apr 11, 2021

@author: immanueltrummer
'''
import argparse
from baselines.common import read_hints, get_parameters, get_values, print_assignments
from collections import defaultdict
import os
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

def predict_keys(sentences):
    """ Returns flags indicating whether sentence is likely key. """
    likely_keys = []
    for s in sentences:
        likely_key = 0
        if get_values(s):
            likely_key = detect_model.predict([s])[0][0]
        likely_keys.append(likely_key)
    return likely_keys

def filter_sentences(sentences):
    """ Returns subset of potentially useful sentences. """
    return list(filter(lambda s: get_parameters(s) or get_values(s), sentences))

if __name__ == '__main__':
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extracts tuning hints via supervised learning')
    parser.add_argument('detect_model_dir', type=str, help='Directory containing detection model')
    parser.add_argument('classify_model_dir', type=str, help='Directory containing classification model')
    parser.add_argument('docs_path', type=str, help='Path of file containing tuning hints')
    parser.add_argument('restrictive', type=str, help='Restrict to specific types of hints')
    args = parser.parse_args()

    # Load models from given directories
    detect_model = ClassificationModel('roberta', args.detect_model_dir, use_cuda=False)
    print('Detection model loaded.')    
    classification_model = ClassificationModel('roberta', args.classify_model_dir, use_cuda=False)
    print('Classification model loaded.')
    
    # Disable warning about tokenizer parallelism
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Collect assignment hints from input documents
    asg_to_doc = defaultdict(lambda: set())
    doc_sentences = read_hints(args.docs_path)
    for doc_id, sentences in doc_sentences.iteritems():
        print(f'Analyzing document nr. {doc_id} ...')
        print(f'Sentences before filtering: {len(sentences)}')
        sentences = filter_sentences(sentences)
        nr_sentences = len(sentences)
        print(f'Sentences after filtering: {nr_sentences}')
        if nr_sentences > 0:
            print('Predicting key sentences')
            #key_prediction = predict_keys(sentences)
            key_prediction = detect_model.predict(sentences)[0]
            nr_keys = sum(key_prediction)
            print(f'\nEstimated number of key sentences: {nr_keys}')
            for i in range(0, nr_sentences):
                if key_prediction[i] == 1:
                    sentence = sentences[i]
                    pred_type = 4
                    if args.restrictive == '1':
                        pred_type = classification_model.predict([sentence])[0][0] 
                    print(f'Predicted hint type: {pred_type}')
                    if pred_type == 4:
                        print(f'Extracting hints from sentence "{sentence}"')
                        values = get_values(sentence)
                        ctx_idx = get_context(sentences, i)
                        ctx_sent = sentences[ctx_idx]
                        params = set(get_parameters(ctx_sent))
                        for param in params:
                            for value in values:
                                assignment = (param, value)
                                asg_to_doc[assignment].add(doc_id)
            # Print extracted assignments after each iteration
            print_assignments(asg_to_doc)