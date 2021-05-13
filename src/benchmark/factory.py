'''
Created on May 12, 2021

@author: immanueltrummer
'''
import benchmark.evaluate

def from_file(config, dbms):
    """ Generate benchmark object from configuration file. 
    
    Args:
        config: describes the benchmark to generate
        dbms: benchmark executed on this DBMS
        
    Returns:
        object representing configured benchmark
    """
    path_to_queries = config['BENCHMARK']['queries']
    path_to_logs = config['BENCHMARK']['logging']
    bench = benchmark.evaluate.OLAP(dbms, path_to_queries)
    bench.reset(path_to_logs)
    return bench