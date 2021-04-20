'''
Created on Apr 17, 2021

@author: immanueltrummer

This module is for debugging.
'''
from doc.collection import DocCollection
from dbms.postgres import PgConfig
from dbms.mysql import MySQLconfig
from all.agents import VQN
from all.approximation import QNetwork
from all.environments import GymEnvironment
from all.experiments import run_experiment
from all.logging import DummyWriter
from all.policies.greedy import GreedyPolicy
from torch.optim import Adam
from torch import nn
from all.presets.classic_control import dqn
from environment.multi_doc import MultiDocTuning
from environment.supervised import LabeledDocTuning
from environment.hybrid import HybridDocTuning
from benchmark.evaluate import OLAP

# Initialize supervised hint extraction environment
labeled_path = '/Users/immanueltrummer/git/literateDBtuners/tuning_docs/pg10docsLabels.csv'
labeled_docs = DocCollection(docs_path=labeled_path, dbms=None, size_threshold=0)
supervised_env = LabeledDocTuning(docs=labeled_docs, nr_hints=500, label_path=labeled_path)

# Initialize unsupervised hint extraction environment
dbms = MySQLconfig('tpch', 'root', 'mysql1234-')

#dbms = PgConfig(db='tpch', user='immanueltrummer')
#unlabeled_docs = DocCollection('/Users/immanueltrummer/git/literateDBtuners/tuning_docs/postgres100', dbms)
unlabeled_docs = DocCollection('/Users/immanueltrummer/git/literateDBtuners/tuning_docs/mysql100', dbms)
benchmark = OLAP(dbms, '/Users/immanueltrummer/git/literateDBtuners/benchmarking/tpch/queries.sql')
unsupervised_env = MultiDocTuning(docs=unlabeled_docs, dbms=dbms, benchmark=benchmark, 
                                  hardware=[2000000, 2000000, 8], nr_hints=100, nr_rereads=50, 
                                  nr_evals=2)

# Initialize hybrid hint extraction environment
hybrid_env = HybridDocTuning(supervised_env, unsupervised_env, 100)
hybrid_env = GymEnvironment(hybrid_env)

# set device
device = 'cpu'

# set writer
writer = DummyWriter()

model = nn.Sequential(
        nn.Linear(1537, 128),
        # nn.ReLU(),
        # nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(128, 5),
    )
def make_model(env):
    return model
    
# create a Pytorch optimizer for the model
optimizer = Adam(model.parameters(), lr=0.01)

# create an Approximation of the Q-function
q = QNetwork(model, optimizer, writer=writer)

# create a Policy object derived from the Q-function
policy = GreedyPolicy(q, 5, epsilon=0.1)

# instantiate the agent
vqn = VQN(q, policy, discount_factor=0.99)

# start experiment
run_experiment(dqn(model_constructor=make_model), hybrid_env, 500)