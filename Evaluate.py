'''
Created on Nov 16, 2020

@author: immanueltrummer
'''
import json
import glob
import os
import psycopg2
import subprocess
import time

def remove_oltp_results():
    """ remove old result files from OLTP benchmark """
    files = glob.glob('/home/ubuntu/oltpbench/oltpbench/'\
                      'results/paramtest*')
    for f in files:
        try:
            os.remove(f)
        except OSError as e:
            print("Error: %s : %s" % (f, e.strerror))

class MySQLeval:
    """ Benchmarks MySQL database """
    def __init__(self, config):
        self.config = config
        
    def change_config(self):
        """ Change configuration via restart, return error flag """
        # Stop MySQL server
        r_code = os.system('sudo /etc/init.d/mysql stop')
        print(f'stopped server with return {r_code}', 
              flush=True)
        # Remove old temporary configuration file
        r_code = os.system('sudo rm tmpConfig')
        print(f'removed configuration file with return {r_code}')
        # Write and copy new configuration file 
        with open('tmpConfig', 'w') as config_file:
            self.config.write(config_file)
        r_code = os.system('sudo cp --remove-destination ' \
                           'tmpConfig /etc/mysql/my.cnf')
        print(f'wrote configuration file with code {r_code}', 
              flush=True)
        # Delete old log files which may prevent server start
        r_code = os.system('sudo rm /var/lib/mysql/ib_logfile*')
        print(f'removed old log files with code {r_code}',
              flush=True)
        # Start server with new configuration
        r_code = os.system('sudo /etc/init.d/mysql start')
        print(f'started server with return {r_code}', 
              flush=True)
        time.sleep(5)
        return r_code != 0
        
    def tpch_eval(self, queries):
        """ Evaluate current configuration with TPC-H """
        # Change configuration and iterate over TPC-H queries
        error = False
        times = [-1 for _ in range(22)]
        try:
            error |= self.change_config()
            print(f"Error status is {error}")
            if not error:
                print("About to run benchmark", flush=True)
                error = True
                for q in queries:
                    start_ms = time.time() * 1000.0
                    r_code = os.system(
                        f'sudo time mysql tpchs1 ' \
                        f'< queries/ms/{q+1}.sql')
                    print(f"Executed with code {r_code}", 
                          flush=True)
                    end_ms = time.time() * 1000.0
                    times[q] = end_ms - start_ms
                error = False
        except Exception as e:
            error = True
            print(f'Exception: {e}')
        return error, times
    
    def copy_db(self, source_dump, target_db):
        """ reload database from reference """
        os.system(f"sudo mysql -e 'drop database if exists {target_db}'")
        print("Dropped old database", flush=True)
        os.system(f"sudo mysql -e 'create database {target_db}'")
        print("Created new database", flush=True)
        os.system(f"sudo mysql {target_db} < {source_dump}")
        print("Initialized new database", flush=True)
        
    def tpcc_eval(self):
        """ evaluates current configuration via benchmark """
        remove_oltp_results()
        throughput = -1
        had_error = True
        try:
            # Change to configuration to evaluate
            self.change_config()
            print('Changed configuration', flush=True)
            # Reload database
            self.copy_db('tpccdump.sql', 'tpcc')
            print('Reloaded database', flush=True)
            # Run benchmark
            return_code = subprocess.run(\
                ['./oltpbenchmark', \
                '-b', 'tpcc', '-c', \
                'firsttest/mysql_tpcc.xml', \
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
    
class PostgresEval():
    """ Benchmark Postgres database """
    def __init__(self, config):
        self.config = config
        
    def change_config(self):
        # Stop Postgres server
        r_code = os.system('sudo pg_ctlcluster 10 main stop')
        print(f'Stopped server with code {r_code}')
        # Override temporary configuration file
        r_code = os.system('sudo rm tmpConfig')
        print(f'removed configuration file with return {r_code}')
        with open('tmpConfig', 'w') as config_file:
            for p in self.config['dummysection']:
                config_file.write(
                    f"{p} = {self.config['dummysection'][p]}\n")
            config_file.flush()
        # Copy temporary configuration file
        r_code = os.system('sudo cp --remove-destination ' \
            'tmpConfig /etc/postgresql/10/main/postgresql.conf')
        print(f'copied configuration file with return {r_code}')
        # Restart Postgres server
        r_code = os.system('sudo pg_ctlcluster 10 main start')
        print(f'Started server with code {r_code}')
        time.sleep(5)
        return r_code != 0
    
    def exec_sql(self, query, dbname, dbuser, dbpassword):
        """ Runs given query on given database """
        connection = psycopg2.connect(database = dbname, 
                user = dbuser, password = dbpassword,
                host = "localhost")
        connection.autocommit = True
        cursor = connection.cursor()
        cursor.execute(query)
        connection.close()
        
    def tpch_eval(self, queries):
        """ Evaluate current configuration with TPC-H """
        # Change configuration and iterate over TPC-H queries
        error = False
        times = [-1 for _ in range(22)]
        try:
            error |= self.change_config()
            print(f"Error status is {error}")
            if not error:
                print("About to run benchmark", flush=True)
                error = True
                for q in queries:
                    start_ms = time.time() * 1000.0
                    with open(f'queries/pg/{q+1}.sql') as q_file:
                        query = q_file.read()
                        self.exec_sql(query, 'tpchs1', 
                                      'postgres', 'postgres')
                    end_ms = time.time() * 1000.0
                    times[q] = end_ms - start_ms
                error = False
        except Exception as e:
            error = True
            print(f'Exception: {e}')
        return error, times
    
    def copy_db(self, source_db, target_db):
        """ reload database from template """
        connection = psycopg2.connect(database = 'postgres', 
                user = 'postgres', password = 'postgres',
                host = 'localhost')
        connection.autocommit = True
        cursor = connection.cursor()
        cursor.execute(f'drop database if exists {target_db};')
        cursor.execute(f'create database {target_db} ' \
                       f'with template {source_db};')
        connection.close()
        
    def tpcc_eval(self):
        """ evaluates current configuration via benchmark """
        remove_oltp_results()
        throughput = -1
        had_error = True
        try:
            # Change to configuration to evaluate
            self.change_config()
            print('Changed configuration', flush=True)
            # Reload database
            self.copy_db('tpcc', 'tpccs2')
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