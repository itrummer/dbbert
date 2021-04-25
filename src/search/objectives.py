'''
Created on Apr 24, 2021

@author: immanueltrummer
'''
from enum import IntEnum

class Objective(IntEnum):
    TIME = 0,  # minimize execution time
    THROUGHPUT = 1,  # maximize throughput
    
def calculate_reward(metrics, default_metrics, objective):
    """ Returns reward metrics, given objectives and metrics. """
    if metrics['error']:
        return -10000
    else:
        if objective == Objective.TIME:
            return default_metrics['time'] - metrics['time']
        elif objective == Objective.THROUGHPUT:
            return metrics['throughput'] - default_metrics['throughput']