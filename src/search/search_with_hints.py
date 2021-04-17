'''
Created on Apr 16, 2021

@author: immanueltrummer
'''
from dbms.generic_dbms import ConfigurableDBMS
from benchmark.evaluate import Benchmark

def search_improvements(dbms: ConfigurableDBMS, benchmark: Benchmark, hint_to_weight, nr_evals):
    """ Try to reduce benchmark time by exploiting weighted hints. Return improvement. """
    return 1