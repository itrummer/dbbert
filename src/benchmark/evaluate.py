'''
Created on Apr 2, 2021

@author: immanueltrummer
'''
from abc import ABC
from abc import abstractmethod
import glob
import os
import pandas as pd
import psycopg2
import subprocess
import time
from dbms.generic_dbms import ConfigurableDBMS

class Benchmark(ABC):
    """ Runs a benchmark to evaluate database configuration. """
    
    @abstractmethod
    def evaluate(self):
        """ Evaluates performance for benchmark and returns reward. """
        pass
    
    @abstractmethod
    def print_stats(self):
        """ Prints out some benchmark statistics. """
        pass
    
    def reset(self, log_path):
        """ Reset timestamps for logging. 
        
        Args:
            log_path: path for logging output
        """
        self.start_ms = time.time() * 1000.0
        self.log_path = log_path
        with open(self.log_path, 'w') as file:
            file.write('millis,quality\n')
        
    def _log(self, quality):
        """ Write quality and timestamp to log file. 
        
        Args:
            quality: quality of best current solution (e.g., w.r.t. throughput)
        """
        with open(self.log_path, 'a') as file:
            cur_ms = time.time() * 1000.0
            total_ms = cur_ms - self.start_ms
            file.write(f'{total_ms},{quality}\n')
    
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
    
    def evaluate(self):
        """ Run all benchmark queries. 
        
        Returns:
            Dictionary containing error flag and time in milliseconds
        """
        self.print_stats()
        start_ms = time.time() * 1000.0
        error = self.dbms.exec_file(self.query_path)
        end_ms = time.time() * 1000.0
        millis = end_ms - start_ms
        # Update statistics
        if not error:
            if millis < self.min_time:
                self.min_time = millis
                self.min_conf = self.dbms.changed() if self.dbms else None
            if millis > self.max_time:
                self.max_time = millis
                self.max_conf = self.dbms.changed() if self.dbms else None
        # Logging
        self._log(self.min_time)
        return {'error': error, 'time': millis}
    
    def print_stats(self):
        """ Print out benchmark statistics. """
        print(f'Minimal time (ms): {self.min_time}')
        print(f'Achieved with configuration: {self.min_conf}')
        print(f'Maximal time (ms): {self.max_time}')
        print(f'Achieved with configuration: {self.max_conf}')
    
class TpcC(Benchmark):
    """ Runs the TPC-C benchmark. """
    
    def __init__(self, oltp_path, config_path, result_path, 
                 dbms, template_db, target_db, reset_every):
        """ Initialize with given paths. 
        
        Args:
            oltp_path: path to OLTP benchmark runner
            config_path: path to configuration file
            result_path: store benchmark results here
            dbms: configurable DBMS (not the benchmark database)
            template_db: used as template to re-initialize DB
            target_db: used for running the benchmark
            reset_every: reset database every i-th evaluation
        """
        self.oltp_path = oltp_path
        self.config_path = config_path
        self.result_path = result_path
        self.dbms = dbms
        self.template_db = template_db
        self.target_db = target_db
        self.reset_every = reset_every
        self.min_throughput = float('inf')
        self.min_config = {}
        self.max_throughput = 0
        self.max_config = {}
        self.evals_since_reset = 0
    
    def _remove_oltp_results(self):
        """ Removes old result files from OLTP benchmark. """
        files = glob.glob(f'{self.result_path}/*')
        for f in files:
            try:
                os.remove(f)
            except OSError as e:
                print("Error: %s : %s" % (f, e.strerror))

    def _reset_db(self):
        """ Reload TPC-C database from template database. """
        self.dbms.copy_db(self.template_db, self.target_db)
        
    def evaluate(self):
        """ Evaluates current configuration on TPC-C benchmark.
        
        Returns:
            Dictionary containing error flag and throughput
         """
        self._remove_oltp_results()
        self.evals_since_reset += 1
        if self.evals_since_reset > self.reset_every:
            self._reset_db()
            self.evals_since_reset = 0
        throughput = -1
        had_error = True
        try:
            # Run benchmark
            return_code = subprocess.run(\
                ['./oltpbenchmark', '-b', 'tpcc', '-c', self.config_path,
                '--execute=true', '-s', '5', '-o', 'tuningtest'],
                cwd = self.oltp_path)
            print(f'Benchmark return code: {return_code}')
            # Extract throughput from generated files
            df = pd.read_csv(f'{self.result_path}/tuningtest.res')
            throughput = df[' throughput(req/sec)'].mean()
            had_error = False
            # Update statistics
            if throughput > self.max_throughput:
                self.max_throughput = throughput
                self.max_config = self.dbms.changed()
            if throughput < self.min_throughput:
                self.min_throughput = throughput
                self.min_config = self.dbms.changed()
        except (Exception, psycopg2.DatabaseError) as e:
            print(e)
        # Logging
        self._log(self.max_throughput)
        return {'error': had_error, 'throughput': throughput}
    
    def print_stats(self):
        """ Print out benchmark statistics. """
        print(f'Minimal throughput {self.min_throughput} with configuration {self.min_config}')
        print(f'Maximal throughput {self.max_throughput} with configuration {self.max_config}')