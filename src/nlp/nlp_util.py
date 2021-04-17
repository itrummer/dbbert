'''
Created on Mar 23, 2021

@author: immanueltrummer
'''
from transformers.models.bert import BertTokenizerFast
from transformers import BertModel
import torch

# Initialize model and associated tokenizer
tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
model = BertModel.from_pretrained("bert-base-cased")

# Initialize caching for natural language analysis
use_cache = False
cached_encodings = {}

# Store cache statistics
cache_hits = 0
cache_misses = 0

def print_cache_stats():
    """ Print statistics for encoding cache. """
    print(f'Nr. cache hits: {cache_hits}')
    print(f'Nr. cache misses: {cache_misses}')

def tokenize(text):
    """ Tokenizes input text using default settings. """
    return tokenizer.encode_plus(
            text, return_offsets_mapping=True, 
            return_tensors="pt", truncation=True)
    
def encode(text):
    """ Generates bidirectional encoding using default settings. """
    global cache_hits
    global cache_misses
    if use_cache and text in cached_encodings:
        encoding = cached_encodings[text]
        cache_hits += 1
    else:
        encoding = model(tokenize(text)['input_ids'])
        cache_misses += 1
        if use_cache:
            cached_encodings[text] = encoding
    return encoding

def mean_encoding(tokens, encoding, start, end):
    """ Returns mean encoding for given character span. """
    # Collect relevant states
    last_states = encoding['last_hidden_state'].squeeze(0).tolist()
    offsets = tokens['offset_mapping'].squeeze(0).tolist()
    m_states = []
    for o, s in zip(offsets, last_states):
        o_start = o[0]
        o_end = o[1]
        if max(start, o_start) <= min(end, o_end):
            m_states.append(s)
    # Truncation may lead to empty states
    if not m_states:
        return None
    else:
        stacked = torch.Tensor(m_states)
        return torch.mean(stacked, dim=0)
    
# def preprocess(doc: DocCollection):
    # """ Encode and cache all passages in the document collection. """
    # for doc_id, passages in enumerate(doc.passages_by_doc):
        # print(f'Treating document nr. {doc_id}')
        # if doc_id != 61:
            # nr_passages = len(passages)
            # for p_id, passage in enumerate(passages):
                # print(f'Encoding passage {p_id}/{nr_passages} (doc. {doc_id})')
                # print(passage)
                # encode(passage)