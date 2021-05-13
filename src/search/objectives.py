'''
Created on Apr 24, 2021

@author: immanueltrummer
'''
from enum import IntEnum

class Objective(IntEnum):
    """ The optimization objective (e.g., latency). """
    
    TIME = 0,  # minimize execution time
    THROUGHPUT = 1,  # maximize throughput
            
def from_file(config):
    """ Parse objective from configuration file. 
    
    Args:
        config: parsed configuration file specifying objective
        
    Returns:
        objective as parsed from file
    """
    obj_str = config['BENCHMARK']['objective']
    if obj_str == 'time':
        return Objective.TIME
    elif obj_str == 'throughput':
        return Objective.THROUGHPUT
    
def calculate_reward(metrics, default_metrics, objective):
    """ Returns reward metrics, given objectives and metrics. """
    if metrics['error']:
        return -10000
    else:
        if objective == Objective.TIME:
            return default_metrics['time'] - metrics['time']
        elif objective == Objective.THROUGHPUT:
            return metrics['throughput'] - default_metrics['throughput']