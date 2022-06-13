'''
Created on Jun 13, 2022

@author: immanueltrummer
'''
import torch

def torch_device():
    """ Returns recommended device. 
    
    Returns:
        GPU device if available, otherwise CPU
    """
    if torch.cuda.is_available():
        return torch.cuda.current_device()
    else:
        return -1