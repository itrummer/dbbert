'''
Created on Apr 30, 2021

@author: immanueltrummer
'''
from doc.collection import DocCollection
from environment.common import DecisionType, DocTuning
from transformers import BertTokenizer
from gym.spaces import Box
import numpy as np
import torch

class TuningBertFine(DocTuning):
    """ Fine-tune BERT to predict action values. """
    
    def __init__(self, docs: DocCollection, hints_per_episode):
        """ Initialize with given document collection. 
        
        Args:
            docs: collection of documents with tuning hints
            hints_per_episode: candidate hints until episode ends
        """
        super().__init__(docs, hints_per_episode)
        self.observation_space = Box(
            low=0, high=100000, shape=(3, 5, 512,), dtype=np.int64)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    def _mask(self, strings, param):
        """ Mask occurrence of parameter in string array. 
        
        Args:
            strings: array of strings
            param: mask this parameter
        """
        return [s.replace(param, '[MASK]') for s in strings]

    def _observe(self):
        """ Generate observation for current decision and hint. """
        if self.hint_ctr < self.nr_hints:
            _, hint = self.hints[self.hint_ctr]
        else:
            _, hint = self.hints[0]
        passage_cps = [hint.passage for _ in range(5)]
        param = hint.param.group()
        value = hint.value.group()
        if self.decision == DecisionType.PICK_BASE:
            choices = [
                f'{param} and {value} relate to main memory.',
                f'{param} and {value} relate to hard disk.',
                f'{param} and {value} relate to core counts.',
                f'Set {param} to {value}.',
                f'{param} and {value} are unrelated.']
        elif self.decision == DecisionType.PICK_FACTOR:
            v_factors = ['much lower than', 'slightly below', 
                         'to', 'slightly above', 'much higher than']
            choices = [f'Set {param} {f} {value}.' for f in v_factors]
        else:
            v_weights = ['not', 'slightly', 'quite', 'very', 'extremely']
            choices = [f'The hint on {param} is {weight} important.' for weight in v_weights]
        # Mask parameter name (generalization to different DBMS)
        passage_cps = self._mask(passage_cps, param)
        choices = self.mask(choices, param)
        encoding = self.tokenizer(
            passage_cps, choices, return_tensors='pt', 
            padding='max_length', truncation=True, max_length=512)
        result = torch.stack((encoding['input_ids'], 
                              encoding['token_type_ids'], 
                              encoding['attention_mask']), dim=0)
        return result