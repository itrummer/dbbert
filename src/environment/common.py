'''
Created on Apr 17, 2021

@author: immanueltrummer
'''
from enum import IntEnum

class DecisionType(IntEnum):
    PICK_BASE=0, # Pick base value for parameter (or decide to neglect hint)
    PICK_FACTOR=1, # Pick a factor to multiply parameter value with
    PICK_WEIGHT=2, # Pick importance of tuning hint