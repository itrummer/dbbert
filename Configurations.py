'''
Created on Sep 6, 2020

@author: immanueltrummer
'''
import json
import numpy
import re
import psycopg2
import time
import subprocess
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
        
    def change_config(self, dbname, dbuser, dbpassword):
        """ Change database configuration """
        self.write_config(self.configPath)
        connection = psycopg2.connect(database = dbname, 
                user = dbuser, password = dbpassword,
                host = "localhost")
        cursor = connection.cursor()
        cursor.execute('SELECT pg_reload_conf();')
        # Wait until configuration has effect
        sleep(3)
        connection.close()

    def evaluateConfig(self, dbname, dbuser, dbpassword):
        """ evaluates current configuration via benchmark """
        self.remove_results()
        self.change_config(dbname, dbuser, dbpassword)
        """ Load configuration for given database and login """
        connection = None
        totalmillis = None
        throughput = None
        error = True
        try:
            # Start time measurements for new configuration
            #startmillis = time.time() * 1000.0
            # Iterate over all queries
            return_code = subprocess.run(\
                ['./oltpbenchmark', \
                '-b', 'tpcc', '-c', \
                'firsttest/sample_tpcc_config.xml', \
                '--execute=true', '-s', '5', \
                '-o', 'paramtest'],\
                cwd = '/home/ubuntu/oltpbench/oltpbench')
            print(f'Return code: {return_code}')
            # Extract throughput from generated files
            with open('/home/ubuntu/oltpbench/oltpbench/results'\
                      '/paramtest.summary') as result_file:
                result_data = json.load(result_file)
                throughput = result_data['Throughput (requests/second)']
                print(f'Throughput: {throughput}')
                result_file.close()
            #totalmillis = time.time() * 1000.0 - startmillis
            # ./oltpbenchmark -b tpcc -c 
            #firsttest/sample_tpcc_config.xml --execute=true -s 5 -o testout3
            """
            #for q in range(1,23):
            for q in range(1,2):
                #q_padded = str(q).zfill(2)
                query_path = f'queries/{q}.sql'
                query = open(query_path, "r").read()
                #print(f'query nr. {q}: {query}')
                cursor.execute(query)
            endmillis = time.time() * 1000.0
            totalmillis = endmillis - startmillis
            result = cursor.fetchone()
            #print(result)
            cursor.close()
            """
            error = False
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        finally:
            if connection is not None:
                connection.close()
                #print("Database connection closed")
        return error, throughput
