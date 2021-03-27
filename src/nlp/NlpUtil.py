'''
Created on Mar 23, 2021

@author: immanueltrummer
'''
import torch
from transformers.models.bert import BertTokenizerFast
from transformers import BertModel

# Initialize model and associated tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
model = BertModel.from_pretrained("bert-base-cased")

# Auxiliary functions for natural language processing
def tokenize(text):
    """ Tokenize using default settings """
    return tokenizer.encode_plus(text, return_offsets_mapping=True, 
                                 return_tensors="pt")
    
def encode(text):
    """ Bidirectional encoding using default settings """
    return model(tokenize(text)['input_ids'])

def word_info(snippet, tokenization, encoded):
    """ Retrieves words with associated encoding. """
    word_ids = [i for i in tokenization.word_ids() if i != None]
    print(word_ids)
    nr_words = max(word_ids) + 1
    h_states = encoded['last_hidden_state'][0]
    # Collect info about each word
    word_info = []
    for word_idx in range(nr_words):
        word_span = tokenization.word_to_chars(word_idx)
        print(word_span)
        word = snippet[word_span[0]:word_span[1]]
        token_idx = tokenization.word_to_tokens(word_idx)
        print(token_idx)
        tokens = h_states[token_idx[0]:token_idx[1]]
        word_info.append((word, tokens))
    return word_info

# s = "Set shared-buffers to 10."
# t = tokenize(s)
# e = encode(s)
# print(word_info(s, t, e))