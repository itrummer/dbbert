'''
Created on May 12, 2021

@author: immanueltrummer
'''
import benchmark.evaluate.OLAP

def from_file(config, dbms=None):
    """ Generate benchmark object from configuration file. 
    
    Args:
        config: describes the benchmark to generate
        dbms: may be 
        
    Returns:
        object representing configured benchmark
    """
    
    return benchmark.evaluate.OLAP(dbms, path_to_queries)