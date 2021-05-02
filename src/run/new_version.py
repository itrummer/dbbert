'''
Created on Apr 27, 2021

@author: immanueltrummer
'''
import gym.spaces
import numpy as np
import torch.nn

from transformers import BertForSequenceClassification
from transformers import BertTokenizer

model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

class NNtest(torch.nn.Module):
    
    def __init__(self):
        super(NNtest, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
        
    def forward(self, observations):
        print(f'Forward shape: {observations.shape}')
        print(observations)
        return self.model(observations).logits

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
input_ids = tokenizer('This is a test.', return_tensors='pt', padding='max_length', truncation=True, max_length=12)['input_ids'][0]

class TestEnv(gym.Env):
    
    def __init__(self):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            low=0, high=100000, shape=(7,), dtype=np.int32)
    
    def _observe(self):
        return input_ids
    
    def step(self, action):
        obs = self._observe()
        reward = 0
        done = True
        return obs, reward, done, {}
    
    def reset(self):
        return self._observe()

from all.environments import GymEnvironment

env = GymEnvironment(TestEnv())
#
from all.agents import VQN
from all.approximation import QNetwork
from all.experiments import run_experiment
from all.logging import DummyWriter
from all.policies.greedy import GreedyPolicy
from torch.optim import Adam
from all.presets.classic_control import dqn

# set device
device = 'cpu'
#
# # set writer
writer = DummyWriter()

model = NNtest()
def make_model(env):
    return model
    
# create a Pytorch optimizer for the model
optimizer = Adam(model.parameters(), lr=0.01)
# # create an Approximation of the Q-function
q = QNetwork(model, optimizer, writer=writer)
# # create a Policy object derived from the Q-function
policy = GreedyPolicy(q, 5, epsilon=0.1)
# # instantiate the agent
vqn = VQN(q, policy, discount_factor=0.99)
# start experiment
run_experiment(dqn(model_constructor=make_model), env, 110)