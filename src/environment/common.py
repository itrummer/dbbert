'''
Created on Apr 17, 2021

@author: immanueltrummer
'''
from abc import ABC, abstractmethod
from doc.collection import DocCollection
from enum import IntEnum
from gym import Env
from gym.spaces import Discrete

class DecisionType(IntEnum):
    """ Describes next decision to make by agent. """
    PICK_BASE=0, # Pick base value for parameter (or decide to neglect hint)
    PICK_FACTOR=1, # Pick a factor to multiply parameter value with
    PICK_WEIGHT=2, # Pick importance of tuning hint
    
class DocTuning(Env, ABC):
    """ Common superclass for environments in which agents tune using text documents. """
    
    def __init__(self, docs: DocCollection, hints_per_episode):
        """ Initialize with given document collection. 
        
        Args:
            docs: collection of documents with tuning hints
            hints_per_episode: candidate hints before evaluation
        """
        self.docs = docs
        self.hints_per_episode = hints_per_episode
        self.action_space = Discrete(5)
        self.factors = [0.25, 0.5, 1, 2, 4]
        self.hint_ctr = 0
    
    def step(self, action):
        """ Potentially apply hint and proceed to next one. """
        reward = self._take_action(action)
        done = self._next_state(action)
        if done:
            reward += self._finalize_episode()
        obs = self._observe()
        return obs, reward, done, {}
    
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
        # Update decision and decide whether to advance
        to_next_hint = False
        if self.decision == DecisionType.PICK_BASE:
            if action == 4: # Skip to next hint
                to_next_hint = True
            else:
                self.decision = DecisionType.PICK_FACTOR
        elif self.decision == DecisionType.PICK_FACTOR:
            self.decision = DecisionType.PICK_WEIGHT
        else:
            to_next_hint = True
        # Did we advance to next hint?
        if to_next_hint:
            self.decision = DecisionType.PICK_BASE
            self.hint_ctr += 1
            if self.hint_ctr >= self.nr_hints:
                self.hint_ctr = 0
            if  self.hint_ctr % self.hints_per_episode == 0:
                return True
        return False
    
    @abstractmethod
    def _observe(self):
        """ Generates observations based on current state. """
        pass
    
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
        self.decision = DecisionType.PICK_BASE
        self.base = None
        self.factor = None
        self._reset()
        obs = self._observe()
        return obs