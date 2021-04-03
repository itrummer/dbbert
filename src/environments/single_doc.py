'''
Created on Mar 22, 2021

@author: immanueltrummer
'''
import gym
import nlp.NlpUtil as nlp
import numpy as np
import pandas as pd
import re
from gym.spaces import Box
from gym.spaces import Discrete
import torch
from dbms import Postgres
from random import randint

class TuningEnv(gym.Env):
    """ Trains agents in understanding tuning hints via NLP. """

    def __init__(self, docs_path):
        """ Initialize from given tuning documents. """
        # Initialize action and observation space
        self.observation_space = Box(
            low=-10, high=10, shape=(1536,), dtype=np.float32)
        self.action_space = Discrete(2)
        # Read tuning hints from file
        self.docs = pd.read_csv(docs_path)
        self.docs.fillna('', inplace=True)
        self.docs = self.docs[self.docs['dbms'] == 'pg']
        #self.nr_docs = self.docs['filenr'].max()
        self.nr_docs = 2
        print('Processing tuning hints')
        self.doc_to_hints = []
        for doc_id in range(self.nr_docs):
            print(doc_id)
            hints = self.doc_hints(doc_id+1)
            print(f'Received {len(hints)} hints for DOC {doc_id}')
            self.doc_to_hints.append(hints)
        # Output a summary of data read
        print('Sample of tuning hints:')
        print(self.docs.sample())
        print(f'Nr. documents read: {self.nr_docs}')
        print(f'Nr. hints read: {self.docs.shape[0]}')
        # Create DBMS configuration interface
        self.dbms = Postgres.PgConfig()
        # Reset environment
        self.reset()
        # Name environment
        #self.name = 'Tuning Environment'
        
    def enrich_nlp(self, matches, tokens, encoding):
        """ Enrich matches using encoder results. """
        enriched = []
        for m in matches:
            m_word = m.group()
            m_start = m.start()
            m_end = m.end()
            # Collect relevant states
            last_states = encoding['last_hidden_state'].squeeze(0).tolist()
            offsets = tokens['offset_mapping'].squeeze(0).tolist()
            m_states = []
            for o, s in zip(offsets, last_states):
                o_start = o[0]
                o_end = o[1]
                if max(m_start, o_start) <= min(m_end, o_end):
                    m_states.append(s)
            stacked = torch.Tensor(m_states)
            mean_state = torch.mean(stacked, dim=0)
            enriched.append((m_word, mean_state))
        return enriched
    
    def snippet_hints(self, snippet):
        """ Generates candidate hints for a text snippet. """
        params = re.finditer(r'[a-z_]+_[a-z]+', snippet)
        values = re.finditer(r'\d+[a-zA-Z]*|on|off', snippet)
        # Enrich mentions with result of NLP analysis
        tokens = nlp.tokenize(snippet)
        encoding = nlp.encode(snippet)
        e_params = self.enrich_nlp(params, tokens, encoding)
        e_values = self.enrich_nlp(values, tokens, encoding)
        # Iterate over parameters and values
        candidates = []
        for param, e_param in e_params:
            print(param)
            for value, e_value in e_values:
                print(value)
                # observation associated with pair
                obs = torch.cat((e_param, e_value))
                candidates.append((param, value, obs))
        return candidates

    def doc_hints(self, doc_id):
        """ Generates all candidate hints for specific doc. """
        snippets_idx = self.docs['filenr'] == doc_id
        snippets = self.docs.loc[snippets_idx, 'sentence']
        doc_hints = []
        for snippet in snippets:
            doc_hints += self.snippet_hints(snippet)
        return doc_hints
        
    def step(self, action):
        """ Potentially apply hint and proceed to next one. """
        # Initialize return values
        reward = 0
        done = False
        # Check for end of episode
        if self.hint_idx >= self.nr_hints:
            done = True
        # Execute action
        if not done and action == 1:
            p, v, _ = self.hints[self.hint_idx]
            success = self.dbms.config(p, v)
            if success:
                reward = 1
            else:
                reward = -1
                done = True
        # Next step unless episode end
        if not done:
            self.hint_idx += 1
        return self.observe(), reward, done, {}
        
    def observe(self):
        """ Returns an observation. """
        if self.hint_idx >= self.nr_hints:
            return torch.zeros(1536)
        else:
            _, _, obs = self.hints[self.hint_idx]
            return obs
        
    def reset(self):
        """ Initializes for new tuning document. """
        doc_id = randint(0, self.nr_docs-1)
        self.hints = self.doc_to_hints[doc_id]
        self.nr_hints = len(self.hints)
        self.hint_idx = 0
        # Initialize DBMS connection
        self.dbms.close_conn()
        self.dbms.create_conn()
        return self.observe()

# pg = Postgres.PgConfig()
# pg.create_conn()
# pg.close_conn()

# print('Testing environment')
# print(os.getcwd())
# env = TuningEnv('../../manuals/AllSentences2.csv')
# c_hints = env.c_hints("A test")
# print(c_hints)

# s = "Set sharedbuffers to 10."
# t = nlp.tokenize(s)
# e = nlp.encode(s)
# print(nlp.word_info(s, t, e))

env = TuningEnv('../../manuals/AllSentences2.csv')

from all.agents import VQN
from all.approximation import QNetwork
from all.environments import GymEnvironment
from all.experiments import run_experiment
from all.logging import DummyWriter
from all.policies.greedy import GreedyPolicy
from torch.optim import Adam
from torch import nn
from all.presets.classic_control import dqn, a2c
from all.environments import GymEnvironment

env = GymEnvironment(env)

# set device
device = 'cpu'

# set writer
writer = DummyWriter()

def make_model(env):
    return nn.Sequential(
        nn.Linear(env.observation_space.shape[0], 128),
        nn.ReLU(),
        nn.Linear(128, env.action_space.n),
    )

model = make_model(env)
#print(model.summary())
    
# create a Pytorch optimizer for the model
optimizer = Adam(model.parameters(), lr=0.01)

# create an Approximation of the Q-function
q = QNetwork(model, optimizer, writer=writer)

# create a Policy object derived from the Q-function
policy = GreedyPolicy(q, env.action_space.n, epsilon=0.1)

# instantiate the agent
vqn = VQN(q, policy, discount_factor=0.99)

# start experiment
run_experiment(dqn(model_constructor=make_model), env, 15000)

#env.dbms.close_conn()
env.close()
#
# for i_episode in range(20):
    # observation = env.reset()
    # for t in range(100):
        # print(t)
        # action = env.action_space.sample()
        # observation, reward, done, info = env.step(action)
        # if done:
            # print("Episode finished after {} timesteps".format(t+1))
            # break
# #env.snippet_hints2('Set shared_buffers to 10')
# env.dbms.close_conn()