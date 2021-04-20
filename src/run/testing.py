'''
Created on Apr 14, 2021

@author: immanueltrummer
'''
from doc.collection import DocCollection
from dbms.postgres import PgConfig
from dbms.mysql import MySQLconfig
from collections import Counter

#dbms = PgConfig(db='tpch', user='immanueltrummer')
dbms = MySQLconfig('tpch', 'root', 'mysql1234-')


#dbms.update('set global innodb_buffer_pool_size=2000000000')
print(dbms.get_value('innodb_buffer_pool_size'))
success = dbms.set_param_smart('innodb_buffer_pool_size', '2GB')
print(success)

#docs = DocCollection('/Users/immanueltrummer/git/literateDBtuners/tuning_docs/postgres100', dbms)

#print('\n'.join(f'{k}, {v}' for k, v in asg_stats.items()))
#
# # Generate corpus of assignment sentences
# # corpus = []
# # for asg in asg_stats:
    # # sentence = f'Set {asg[0]} to {asg[1]}.'
    # # corpus.append(sentence)
# params = Counter()
# for asg in asg_stats:
    # if dbms.is_param(asg[0]):
        # params.update([asg[0]])
        #
# print('\n'.join(f'{k}, {v}' for k, v in params.most_common()))
# print('\n'.join(f'{k}, {v}' for k, v in asg_stats.most_common()))
#
# corpus = []
# for param in params:
    # if dbms.is_param(param):
        # corpus.append(str(param))
        #
# """
# This is a simple application for sentence embeddings: clustering
# Sentences are mapped to sentence embeddings and then k-mean clustering is applied.
# """
# from sentence_transformers import SentenceTransformer
# from sklearn.cluster import KMeans
#
# embedder = SentenceTransformer('paraphrase-distilroberta-base-v1')
#
# # Corpus with example sentences
# # corpus = ['A man is eating food.',
          # # 'A man is eating a piece of bread.',
          # # 'A man is eating pasta.',
          # # 'The girl is carrying a baby.',
          # # 'The baby is carried by the woman',
          # # 'A man is riding a horse.',
          # # 'A man is riding a white horse on an enclosed ground.',
          # # 'A monkey is playing drums.',
          # # 'Someone in a gorilla costume is playing a set of drums.',
          # # 'A cheetah is running behind its prey.',
          # # 'A cheetah chases prey on across a field.'
          # # ]
# corpus_embeddings = embedder.encode(corpus)
#
# # Perform kmean clustering
# num_clusters = 3
# clustering_model = KMeans(n_clusters=num_clusters)
# clustering_model.fit(corpus_embeddings)
# cluster_assignment = clustering_model.labels_
#
# clustered_sentences = [[] for i in range(num_clusters)]
# for sentence_id, cluster_id in enumerate(cluster_assignment):
    # clustered_sentences[cluster_id].append(corpus[sentence_id])
    #
# for i, cluster in enumerate(clustered_sentences):
    # print("Cluster ", i+1)
    # print(cluster)
    # print("")