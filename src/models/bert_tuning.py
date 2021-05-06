'''
Created on May 5, 2021

@author: immanueltrummer

Agent models, based on BERT.
'''
from transformers import BertForMultipleChoice
import torch.nn

class BertFineTuning(torch.nn.Module):
    """ Used for Q learning in the fine-tuning environment. """
    def __init__(self):
        """ Initialize for cased text. """
        super(BertFineTuning, self).__init__()
        self.model = BertForMultipleChoice.from_pretrained('bert-base-cased')
        
    def forward(self, observations):
        """ Produces logits for five actions. """
        permuted_obs = observations.permute(1, 0, 2, 3).contiguous()
        return self.model(input_ids=permuted_obs[0], 
                          token_type_ids=permuted_obs[1],
                          attention_mask=permuted_obs[2]).logits