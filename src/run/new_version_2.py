'''
Created on May 1, 2021

@author: immanueltrummer
'''
from doc.collection import DocCollection
from environment.supervised import LabeledDocTuning
from all.environments import GymEnvironment
from transformers import BertForMultipleChoice
import torch.nn

labeled_path = '/Users/immanueltrummer/git/literateDBtuners/tuning_docs/pg10docsLabels.csv'
labeled_docs = DocCollection(docs_path=labeled_path, dbms=None, size_threshold=0)
supervised_env = LabeledDocTuning(docs=labeled_docs, nr_hints=10, label_path=labeled_path)
supervised_env = GymEnvironment(supervised_env)

class NNtest(torch.nn.Module):
    
    def __init__(self):
        super(NNtest, self).__init__()
        self.model = BertForMultipleChoice.from_pretrained('bert-base-uncased')
        
    def forward(self, observations):
        permuted_obs = observations.permute(1, 0, 2, 3).contiguous()
        return self.model(input_ids=permuted_obs[0], 
                          token_type_ids=permuted_obs[1],
                          attention_mask=permuted_obs[2]).logits

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
run_experiment(dqn(model_constructor=make_model, minibatch_size=2), supervised_env, 5000)