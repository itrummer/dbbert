'''
Created on Apr 17, 2021

@author: immanueltrummer
'''
from abc import ABC, abstractmethod
from doc.collection import DocCollection, TuningHint
from enum import IntEnum
from gym import Env
from gym.spaces import Box, Discrete
import nlp.nlp_util as nlp
import numpy as np
import torch

class DecisionType(IntEnum):
    PICK_BASE=0, # Pick base value for parameter (or decide to neglect hint)
    PICK_FACTOR=1, # Pick a factor to multiply parameter value with
    PICK_WEIGHT=2, # Pick importance of tuning hint
    
class DocTuning(Env, ABC):
    """ Common superclass for environments in which agents tune using text documents. """
    
    def __init__(self, docs: DocCollection):
        """ Initialize with given document collection. 
        
        Args:
            docs: collection of documents with tuning hints
        """
        self.docs = docs
        self.observation_space = Box(
            low=-10, high=10, shape=(1537,), dtype=np.float32)
        self.action_space = Discrete(5)
        self.def_obs = torch.zeros(1537)
        self.def_hint_obs = torch.zeros(1536)
        self.obs_cache = {}
        self.nr_rereads = 1
        self.factors = [0.25, 0.5, 1, 2, 4]
    
    def step(self, action):
        """ Potentially apply hint and proceed to next one. """
        reward = self._take_action(action)
        done = self._next_state(action)
        if done:
            reward += self._finalize_episode()
        return self._observe(), reward, done, {}
    
    @abstractmethod
    def _take_action(self, action):
        """ Process action and return obtained reward. """
        pass
     
    @abstractmethod
    def _finalize_episode(self):
        """ Finalize current episode and return reward. """
        pass
     
    def _next_state(self, action):
        """ Advance to next state in MDP and return termination flag. """
        done = False
        # Update decision and hint counter
        if self.decision == DecisionType.PICK_BASE:
            if action == 4: # Skip to next hint
                self.hint_ctr += 1
            else:
                self.decision = DecisionType.PICK_FACTOR
        elif self.decision == DecisionType.PICK_FACTOR:
            self.decision = DecisionType.PICK_WEIGHT
        else:
            self.decision = DecisionType.PICK_BASE
            self.hint_ctr += 1
        # Update rereads and termination flag
        if self.hint_ctr >= self.nr_hints:
            self.reread_ctr += 1
        if self.reread_ctr >= self.nr_rereads:
            done = True
        return done
    
    def _observe(self):
        """ Generates observations based on current hint. """
        obs = self.def_obs
        if self.hint_ctr < self.nr_hints:
            if self.hint_ctr in self.obs_cache:
                obs = self.obs_cache[self.hint_ctr]
            else:
                _, hint = self.hints[self.hint_ctr]
                hint_obs = self._hint_to_obs(hint)
                decision_obs = torch.tensor([int(self.decision)])
                obs = torch.cat((hint_obs, decision_obs))
                self.obs_cache[self.hint_ctr] = obs
        return obs

    def _hint_to_obs(self, hint: TuningHint):
        """ Maps tuning hint to an observation vector. """
        tokens = nlp.tokenize(hint.passage)
        encoding = nlp.encode(hint.passage)
        # Map parameter and value to vector
        obs_parts = []
        for item in [hint.param, hint.value]:
            obs_parts.append(
                nlp.mean_encoding(
                    tokens, encoding, item.start(), item.end()))
        # Use zeros in case of missing vectors
        obs = self.def_hint_obs
        if not (obs_parts[0] is None or obs_parts[1] is None):
            obs = torch.cat((obs_parts[0], obs_parts[1]))
        return obs
    
    @abstractmethod
    def _reset(self):
        """ Specialized reset by a sub-class. 
        
        E.g., this may set up the following variables:
        - self.hints - currently considered hints
        - self.nr_hints - number of current hints
        """
        pass
    
    def reset(self):
        """ Initializes for new tuning episode. """
        self.reread_ctr = 0
        self.hint_ctr = 0
        self.decision = DecisionType.PICK_BASE
        self.base = None
        self.factor = None
        self._reset()
        obs = self._observe()
        return obs