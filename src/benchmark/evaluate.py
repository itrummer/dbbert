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
from dbms.configurable_dbms import ConfigurableDBMS

class Benchmark(ABC):
    """ Runs a benchmark to evaluate database configuration. """
    
    @abstractmethod
    def evaluate(self):
        """ Evaluates performance for benchmark and returns reward. """
        pass
    
class TpcH(Benchmark):
    """ Runs the TPC-H benchmark. """
    
    def __init__(self, dbms: ConfigurableDBMS, queries):
        """ Initialize with DBMS engine and IDs of queries to consider. """
        self.dbms = dbms
        self.queries = queries
    
    def evaluate(self):
        """ Run selected TPC-H queries. 
        
        Returns:
            Boolean error flag and time in milliseconds
        """
        error = True
        start_ms = time.time() * 1000.0
        try:
            # Iterate over selected queries
            for q in self.queries:
                with open(f'queries/pg/{q+1}.sql') as q_file:
                    query = q_file.read()
                    self.dbms.query(query)
            # Set error flag to False
            error = False
        except Exception as e:
            print(f'Exception: {e}')
        # Measure total time in milliseconds
        end_ms = time.time() * 1000.0
        millis = end_ms - start_ms
        return error, millis
    
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