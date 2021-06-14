'''
Created on May 5, 2021

@author: immanueltrummer

Agent models, based on BERT.
'''
from transformers import trainer_utils
from transformers import BertForMultipleChoice
import torch.nn

class BertFineTuning(torch.nn.Module):
    """ Used for Q learning in the fine-tuning environment. """
    def __init__(self, start_model):
        """ Initialize for cased text. 
        
        Args:
            start_model: name or path to pretrained model
        """
        super(BertFineTuning, self).__init__()
        trainer_utils.set_seed(42)
        self.model = BertForMultipleChoice.from_pretrained(start_model)
        
    def forward(self, observations):
        """ Produces logits for five actions. """
        permuted_obs = observations.permute(1, 0, 2, 3).contiguous()
        return self.model(input_ids=permuted_obs[0], 
                          token_type_ids=permuted_obs[1],
                          attention_mask=permuted_obs[2]).logits