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
    bench_type = config['BENCHMARK']['type']
    if bench_type == 'olap':
        path_to_queries = config['BENCHMARK']['queries']
        bench = benchmark.evaluate.OLAP(dbms, path_to_queries)
    else:
        template_db = config['DATABASE']['template_db']
        target_db = config['DATABASE']['target_db']
        oltp_home = config['BENCHMARK']['oltp_home']
        oltp_config = config['BENCHMARK']['oltp_config']
        oltp_result = config['BENCHMARK']['oltp_result']
        reset_every = int(config['BENCHMARK']['reset_every'])
        bench = benchmark.evaluate.TpcC(
            oltp_home, oltp_config, oltp_result, 
            dbms, template_db, target_db, reset_every)
    return bench