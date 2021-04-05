'''
Created on Mar 22, 2021

@author: immanueltrummer
'''
import gym
import nlp.nlp_util as nlp
import numpy as np
from gym.spaces import Box
from gym.spaces import Discrete
import torch
from random import randint
from termcolor import cprint
from doc.collection import DocCollection, TuningHint
from dbms.generic_dbms import ConfigurableDBMS

class TuningEnv(gym.Env):
    """ Trains agents in understanding tuning hints via NLP. """

    def __init__(self, docs: DocCollection, dbms: ConfigurableDBMS):
        """ Initialize from given tuning documents. """
        self.observation_space = Box(low=-10, high=10, 
                                     shape=(1536,), dtype=np.float32)
        self.action_space = Discrete(2)
        self.docs = docs
        self.dbms = dbms
        self.obs_cache = {}
        self.p_vals = set()
        self.reset()
        
    def step(self, action):
        """ Potentially apply hint and proceed to next one. """
        # Initialize return values
        reward = 1
        done = False
        # Check for end of episode
        if self.hint_idx >= self.nr_hints:
            done = True
        # Execute action
        if not done and action == 1:
            hint: TuningHint = self.hints[self.hint_idx]
            param = hint.param.group()
            value = hint.value.group()
            success = self.dbms.can_set(param, value, 1)
            if success:
                #quoted_value = '\'' + value + '\'' if hint.quotes else value
                output = f'Set {param} to {value}!'
                self.p_vals.add((param, value))
                print(output)
                reward = 5
            else:
                reward = -10
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
            index = (self.doc_id, self.hint_idx)
            if index in self.obs_cache:
                return self.obs_cache[index]
            else:
                hint: TuningHint = self.hints[self.hint_idx]
                tokens = nlp.tokenize(hint.passage)
                encoding = nlp.encode(hint.passage)
                obs_parts = []
                for item in [hint.param, hint.value]:
                    obs_parts.append(nlp.mean_encoding(tokens, encoding, 
                                                       item.start(), item.end()))
                # Check for empty results
                obs = torch.zeros(1536)
                if not (obs_parts[0] is None or obs_parts[1] is None):
                    obs = torch.cat((obs_parts[0], obs_parts[1]))
                self.obs_cache[index] = obs
                return obs
        
    def reset(self):
        """ Initializes for new tuning document. """
        self.doc_id = randint(0, self.docs.nr_docs-1)
        print(f'Selected document nr. {self.doc_id}')
        self.hints = self.docs.get_hints(self.doc_id)
        self.nr_hints = len(self.hints)
        self.hint_idx = 0
        self.dbms.reset_config()
        for p_val in self.p_vals:
            print(f'{p_val}\n')
        return self.observe()
