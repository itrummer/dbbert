'''
Created on Sep 6, 2020

@author: immanueltrummer
'''
import json
import numpy
import re
import psycopg2
import subprocess
import time
import os
import glob

from sentence_transformers import SentenceTransformer
from time import sleep

# Initialize NLP via BERT
model = SentenceTransformer('bert-base-nli-mean-tokens')

class TuningParam:
    """ Represents one line in the configuration file """

    def __init__(self, line):
        """ Initialize from original line """
        self.sline = line.strip("#").rstrip("\n")
        self.tokens = re.split("_| |\=|\\t|#|kB|MB|GB|TB", self.sline)
        print(self.tokens)
        self.name = self.sline.split()[0]
        print(f'Parameter name is {self.name}')
        self.numbers = list(filter(lambda x : x.isdigit(), self.tokens))
        print(self.numbers)
        self.isNumeric = True if self.numbers else False
        print(self.isNumeric)
        self.sentence = " ".join(self.tokens)
        print(self.sentence)
        self.embedding = model.encode([self.sentence])[0]
        #print(self.embedding)
        print(self.embedding.shape)

    def scaled_line(self, factor):
        """ Outputs associated configuration line scaled by factor """
        scaledLine = self.sline
        if factor != 1:
            for nr in self.numbers:
                intNr = int(nr)
                scaledIntNr = int(factor * intNr)
                scaledNr = str(scaledIntNr)
                scaledLine = scaledLine.replace(nr, scaledNr)
        return scaledLine

class TuningConfig:
    """ Represents configuration file and offers associated functions """
    def __init__(self, path):
        """ Reads tunable parameters from tuning file """
        print("Initializing tuning configuration")
        self.configPath = path # path to tuning file
        self.nrLines = 0 # counts number of lines in file
        self.idToLine = {} # maps line ID to line string
        self.idToTunable = {} # maps line ID to parameter
        self.idToFactor = {} # maps line ID to scaling factor
        with open(path, encoding="utf-8") as file:
            for line in file:
                print(line)
                self.nrLines += 1
                lineID = self.nrLines - 1
                self.idToLine[lineID] = line
                if ("=" in line):
                    param = TuningParam(line)
                    if (param.isNumeric):
                        print("Numeric parameter")
                        self.idToTunable[lineID] = param
                        self.idToFactor[lineID] = 1
        self.nr_evaluations = 0 # number of evaluations

    def set_scale(self, lineID, factor):
        """ Set scaling factor for parameter """
        self.idToFactor[lineID] = factor
        
    def get_factors(self):
        """ Returns vector of all factors """
        factors = numpy.ones(self.nrLines)
        for line_id, factor in self.idToFactor.items():
            factors[line_id] = factor
        return factors
            
    def load_factors(self, factors):
        """ Initializes for given factors """
        for line in range(self.nrLines):
            factor = factors[line]
            self.idToFactor[line] = factor 

    def restore_defaults(self):
        """ Reset all scaling factors to defaults """
        for lineID in self.idToTunable.keys():
            self.idToFactor[lineID] = 1
    
    def remove_results(self):
        files = glob.glob('/home/ubuntu/oltpbench/oltpbench/'\
                          'results/paramtest*')
        for f in files:
            try:
                os.remove(f)
            except OSError as e:
                print("Error: %s : %s" % (f, e.strerror))
    
    def write_config(self, path):
        """ Write configuration with scaled parameters to file """
        f = open(path, "w")
        for lineID in range(0, self.nrLines):
            if lineID in self.idToTunable:
                param = self.idToTunable[lineID]
                factor = self.idToFactor[lineID]
                f.write(f'# With factor {factor}:\n')
                f.write(param.scaled_line(factor))
                f.write("\n")
            else:
                f.write(self.idToLine[lineID])
        f.close()
        
    def exec_sql(self, query, dbname, dbuser, dbpassword):
        """ Runs given query on given database """
        connection = psycopg2.connect(database = dbname, 
                user = dbuser, password = dbpassword,
                host = "localhost")
        connection.autocommit = True
        cursor = connection.cursor()
        cursor.execute(query)
        connection.close()
        
    def change_config(self):
        """ Change database configuration """
        self.write_config(self.configPath)
        # Restart database server
        os.system('sudo sh -c "pg_ctlcluster 10 main restart"')
        """
        code = subprocess.run(['sudo', 'sh', '-c', 
                               '"pg_ctlcluster', 
                               '10', 'main', 'restart"'])
        print(f'Restarted server - return code {code}')
        """
        print("Restarted server")
        
    def reload_db(self):
        """ reload TPC-C database from template """
        connection = psycopg2.connect(database = 'postgres', 
                user = 'postgres', password = 'postgres',
                host = 'localhost')
        connection.autocommit = True
        cursor = connection.cursor()
        cursor.execute('drop database if exists tpccs2;')
        cursor.execute('create database tpccs2 with template tpcc;')
        connection.close()
        
    def tpch_eval(self):
        """ Evaluate current configuration with TPC-H """
        self.change_config()
        # Iterate over TPC-H queries
        error = True
        total_ms = -1
        start_ms = time.time() * 1000.0
        try:
            for q in range(1,23):
                query_path = f'queries/{q}.sql'
                query = open(query_path, "r").read()
                self.exec_sql(query, 'tpchs1oltp', 
                              'postgres', 'postgres')
            end_ms = time.time() * 1000.0
            total_ms = end_ms - start_ms
            error = False
        except Exception as e:
            print(e)
        return error, total_ms

    def evaluateConfig(self, dbname, dbuser, dbpassword):
        """ evaluates current configuration via benchmark """
        self.nr_evaluations += 1
        self.remove_results()
        throughput = None
        error = True
        try:
            # Reload database
            self.reload_db()
            print('Reloaded database')
            # Change to configuration to evaluate
            self.change_config()
            print('Changed configuration')
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
                throughput = result_data['Throughput (requests/second)']
                print(f'Throughput: {throughput}')
                result_file.close()
            error = False
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        return error, throughput
