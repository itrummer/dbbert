'''
Created on Nov 11, 2020

@author: immanueltrummer
'''
""" Analyze impact of tuning parameters. """

import configparser
import Configurations
import Evaluate

# Prepare tuning and perform NLP analysis
"""
print("About to create Postgres tuning config")
pg_config = Configurations.TuningConfig(
    '/etc/postgresql/10/main/postgresql.conf')
"""
print("Reading default MySQL configuration file")
config = configparser.ConfigParser()
config.read('mysql/all_tunable_config.cnf')
print("Create MySQL evaluation object")
tpch_eval = Evaluate.MySQLeval(config)

# Iterate over all parameters and try different settings
f = open("mysqlTpchAnalysis.txt", "w")
for param in config['mysqld-5.7']:
    # Extract original parameter value
    val = config['mysqld-5.7'][param]
    print(f"Parameter {param} with value {val}")
    # Iterate over alternative values
    for alt_val in Configurations.alternative_vals(val):
        config['mysqld-5.7'][param] = alt_val
        print(f'Trying alternative value {alt_val}')
        error, millis = tpch_eval.tpch_eval()
        f.write(f'{param}\t{alt_val}\t{error}\t{millis}\n')
        f.flush()
f.close()

"""
f = open("tpchAnalysis.txt", "w")
for lineID, param in pg_config.idToTunable.items():
    for factor in (0.2, 5, 1):
        pg_config.set_scale(lineID, factor)
        print(f'Trying with factor {factor} for {param.name}')
        error, millis = pg_config.tpch_eval()
        print(f'Trying with factor {factor}')
        f.write(f'{param.name}\t{factor}\t{error}\t{millis}\n')
        f.flush()
f.close()
"""