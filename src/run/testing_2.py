'''
Created on May 7, 2021

@author: immanueltrummer
'''
from sentence_transformers import SentenceTransformer, util

#model = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
model = SentenceTransformer('paraphrase-distilroberta-base-v1')


# Single list of sentences - Possible tens of thousands of sentences

import configparser
config = configparser.ConfigParser()
with open('/Users/immanueltrummer/git/literateDBtuners/config/pg13.conf') as stream:
    config.read_string("[dummysection]\n" + stream.read())

# for p in config['dummysection']:
    # print(p)
    
print(list(config['dummysection']))

sentences = [
        'As in our previous experiments, we disable nested loop joins when the inner relation is not an index lookup (i.e., non-index nested loop joins).',
    *list(config['dummysection']), 
    'enable_nest_loop']

embeddings = model.encode(sentences, convert_to_tensor=True)
cosine_scores = util.pytorch_cos_sim(embeddings, embeddings)

pairs = [(sentences[i], cosine_scores[0][i]) for i in range(len(sentences))]
pairs.sort(key=lambda x:x[1])

print(pairs)
#
# for i in range(len(sentences)):
    # print(f'{sentences[i]}, {sentences[0]} {cosine_scores[0][i]}')
#
# paraphrases = util.paraphrase_mining(model, sentences)
#
# for paraphrase in paraphrases[1000:2000]:
    # score, i, j = paraphrase
    # print("{} \t\t {} \t\t Score: {:.4f}".format(sentences[i], sentences[j], score))