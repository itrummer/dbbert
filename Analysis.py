'''
Created on Nov 11, 2020

@author: immanueltrummer
'''
""" Analyze impact of tuning parameters. """

import configparser
import Evaluate

def alternative_vals(val):
    """ Returns alternative parameter values to try. 
        The original value is returned as last element. """
    alt_vals = []
    if val.isdigit():
        alt_vals += [str(int(int(val) * 0.2)), 
                     str(int(val) * 5)]
    elif val.lower() == 'on':
        alt_vals += ['off']
    elif val.lower() == 'off':
        alt_vals += ['on']
    return alt_vals + [val]

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
    print(f"Parameter {param} with value {val}", flush=True)
    # Iterate over alternative values
    alt_vals = alternative_vals(val)
    print(f"Alternative values: {alt_vals}", flush=True)
    if len(alt_vals) > 1:
        for alt_val in alt_vals:
            config['mysqld-5.7'][param] = alt_val
            print(f'Alternative {alt_val} to {val}', flush=True)
            error, millis = tpch_eval.tpch_eval()
            f.write(f'{param}\t{alt_val}\t{error}\t{millis}\n')
            f.flush()
f.close()