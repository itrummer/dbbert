'''
Created on Apr 2, 2021

@author: immanueltrummer
'''
from abc import ABC
from abc import abstractmethod
import glob
import json
import os
import psycopg2
import subprocess
import time
from dbms.generic_dbms import ConfigurableDBMS

class Benchmark(ABC):
    """ Runs a benchmark to evaluate database configuration. """
    
    @abstractmethod
    def evaluate(self, dbms: ConfigurableDBMS = None):
        """ Evaluates performance for benchmark and returns reward. """
        pass
    
    @abstractmethod
    def print_stats(self):
        """ Prints out some benchmark statistics. """
        pass
    
class OLAP(Benchmark):
    """ Runs an OLAP style benchmark with single queries stored in files. """
    
    def __init__(self, dbms: ConfigurableDBMS, query_path):
        """ Initialize with database and path to queries. """
        self.dbms = dbms
        self.query_path = query_path
        self.min_time = float('inf')
        self.max_time = 0
        self.min_conf = {}
        self.max_conf = {}
    
    def evaluate(self, dbms: ConfigurableDBMS = None):
        """ Run all benchmark queries. 
        
        Returns:
            Boolean error flag and time in milliseconds
        """
        start_ms = time.time() * 1000.0
        error = self.dbms.exec_file(self.query_path)
        end_ms = time.time() * 1000.0
        millis = end_ms - start_ms
        # Update statistics
        if not error:
            if millis < self.min_time:
                self.min_time = millis
                self.min_conf = dbms.changed() if dbms else None
            if millis > self.max_time:
                self.max_time = millis
                self.max_conf = dbms.changed() if dbms else None 
        return error, millis
    
    def print_stats(self):
        """ Print out benchmark statistics. """
        print(f'Minimal time (ms): {self.min_time}')
        print(f'Achieved with configuration: {self.min_conf}')
        print(f'Maximal time (ms): {self.max_time}')
        print(f'Achieved with configuration: {self.max_conf}')
    
# TODO: replace hard-coded paths and database names
class TpcC(Benchmark):
    """ Runs the TPC-C benchmark. """
    
    def _remove_oltp_results(self):
        """ Removes old result files from OLTP benchmark. """
        files = glob.glob('/home/ubuntu/oltpbench/oltpbench/results/paramtest*')
        for f in files:
            try:
                os.remove(f)
            except OSError as e:
                print("Error: %s : %s" % (f, e.strerror))

    def _copy_db(self, source_db, target_db):
        """ Reload TPC-C database from template database. """
        connection = psycopg2.connect(database = 'postgres', 
                user = 'postgres', password = 'postgres',
                host = 'localhost')
        connection.autocommit = True
        cursor = connection.cursor()
        cursor.execute(f'drop database if exists {target_db};')
        cursor.execute(f'create database {target_db} ' \
                       f'with template {source_db};')
        connection.close()

    def evaluate(self):
        """ Evaluates current configuration on TPC-C benchmark.
        
        Returns:
            Boolean error flag and throughput
         """
        self._remove_oltp_results()
        throughput = -1
        had_error = True
        try:
            # Change to configuration to evaluate
            self.change_config()
            print('Changed configuration', flush=True)
            # Reload database
            self._copy_db('tpcc', 'tpccs2')
            print('Reloaded database', flush=True)
            # Run benchmark
            return_code = subprocess.run(\
                ['./oltpbenchmark', \
                '-b', 'tpcc', '-c', \
                'firsttest/sample_tpcc_config.xml', \
                '--execute=true', '-s', '5', \
                '-o', 'paramtest'],\
                cwd = '/home/ubuntu/oltpbench/oltpbench')                
            print(f'Benchmark return code: {return_code}')
            # Extract throughput from generated files
            with open('/home/ubuntu/oltpbench/oltpbench/results'\
                      '/paramtest.summary') as result_file:
                result_data = json.load(result_file)
                throughput = result_data[
                    'Throughput (requests/second)']
                print(f'Throughput: {throughput}', flush=True)
                result_file.close()
            had_error = False
        except (Exception, psycopg2.DatabaseError) as e:
            print(e)
        return had_error, throughput