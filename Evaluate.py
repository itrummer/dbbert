'''
Created on Nov 16, 2020

@author: immanueltrummer
'''
import os
import time

class MySQLeval:
    """ Benchmarks MySQL database """
    def __init__(self, config):
        self.config = config
        
    def change_config(self):
        """ Change configuration via restart """
        self.config.override_config()
        os.system('sudo /etc/init.d/mysql restart')
        
    def tpch_eval(self):
        """ Evaluate current configuration with TPC-H """
        # Change configuration and iterate over TPC-H queries
        error = True
        total_ms = -1
        start_ms = time.time() * 1000.0
        try:
            self.config.change_config()
            for q in range(1,2):
                os.system(f'sudo mysql tpchs1oltp ' \
                          f'< queries/{q}.sql')
            end_ms = time.time() * 1000.0
            total_ms = end_ms - start_ms
            error = False
        except Exception as e:
            print(e)
        return error, total_ms