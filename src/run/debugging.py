'''
Created on Apr 17, 2021

@author: immanueltrummer

This module is for debugging.
'''
from doc.collection import DocCollection
from dbms.postgres import PgConfig
from all.agents import VQN
from all.approximation import QNetwork
from all.environments import GymEnvironment
from all.experiments import run_experiment
from all.logging import DummyWriter
from all.policies.greedy import GreedyPolicy
from torch.optim import Adam
from torch import nn
from all.presets.classic_control import dqn
from environment.supervised import LabeledDocTuning
from benchmark.evaluate import OLAP
from dbms.mysql import MySQLconfig
from dbms.postgres import PgConfig


docs = DocCollection('/Users/immanueltrummer/git/literateDBtuners/tuning_docs/pg10docs.csv')
env = LabeledDocTuning(docs, 10)
env = GymEnvironment(env)

# set device
device = 'cpu'

# set writer
writer = DummyWriter()

def make_model(env):
    return nn.Sequential(
        nn.Linear(env.observation_space.shape[0], 128),
        # nn.ReLU(),
        # nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(128, env.action_space.n),
    )
model = make_model(env)
    
# create a Pytorch optimizer for the model
optimizer = Adam(model.parameters(), lr=0.01)

# create an Approximation of the Q-function
q = QNetwork(model, optimizer, writer=writer)

# create a Policy object derived from the Q-function
policy = GreedyPolicy(q, env.action_space.n, epsilon=0.1)

# instantiate the agent
vqn = VQN(q, policy, discount_factor=0.99)

# start experiment
run_experiment(dqn(model_constructor=make_model), env, 10000)