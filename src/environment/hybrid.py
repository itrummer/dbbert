'''
Created on Apr 19, 2021

@author: immanueltrummer
'''
from environment.multi_doc import MultiDocTuning
from environment.supervised import LabeledDocTuning
from gym import Env
from gym.spaces import Box, Discrete
import numpy as np

class HybridDocTuning(Env):
    """ Mix of supervised and unsupervised tuning. """
    
    def __init__(self, supervised_env: LabeledDocTuning, 
                 unsupervised_env: MultiDocTuning, switch_after):
        """ Initialize for mix of supervised and unsupervised learning. 
        
        Args:
            supervised_env: supervised extraction of tuning hints.
            unsupervised_env: unsupervised extraction of tuning hints.
            switch_after: switch to unsupervised learning after so many episodes.
        """
        self.supervised_env = supervised_env
        self.unsupervised_env = unsupervised_env
        self.switch_after = switch_after
        self.episode_ctr = 0
        self.current_env = supervised_env
        self.observation_space = Box(
            low=-10, high=10, shape=(1537,), dtype=np.float32)
        self.action_space = Discrete(5)
        
    def step(self, action):
        return self.current_env.step(action)
        
    def reset(self):
        # Update current environment
        self.episode_ctr += 1
        if self.episode_ctr > self.switch_after:
            self.current_env = self.unsupervised_env
        # Invoke associated reset function
        return self.current_env.reset()