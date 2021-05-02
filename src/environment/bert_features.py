'''
Created on Apr 30, 2021

@author: immanueltrummer
'''
from doc.collection import DocCollection, TuningHint
from environment.common import DocTuning
from gym.spaces import Box
import nlp.nlp_util as nlp
import numpy as np
import torch

class TuningBertFeatures(DocTuning):
    """ Uses BERT to extract features for hint classification. """
    
    def __init__(self, docs: DocCollection):
        """ Initialize with given document collection. 
        
        Args:
            docs: collection of documents with tuning hints
        """
        super().__init__(docs)
        self.observation_space = Box(
            low=-10, high=10, shape=(1538,), dtype=np.float32)
        self.def_obs = torch.zeros(1538)
        self.def_hint_obs = torch.zeros(1537)
        self.obs_cache = {}
    
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
            doc_obs = torch.tensor([int(hint.doc_id)])
            obs = torch.cat((obs_parts[0], obs_parts[1], doc_obs))
        return obs