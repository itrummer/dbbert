'''
Created on Nov 11, 2020

@author: immanueltrummer
'''
""" Analyze impact of tuning parameters. """

import configparser
import Evaluate
import re

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
    elif re.match("[0-9]+.*", val):
        # Numbers followed by unit
        m = re.match("([0-9]+)(.*)", val)
        nr = m.group(1)
        unit = m.group(2)
        alt_vals += [
            str(int(int(nr) * 0.2)) + unit, 
            str(int(int(nr) * 5)) + unit]
    return alt_vals + [val]

# Prepare tuning for MySQL
print("Reading default MySQL configuration file")
ms_config = configparser.ConfigParser()
ms_config.read('mysql/all_tunable_config.cnf')
print("Creating MySQL evaluation object")
ms_eval = Evaluate.MySQLeval(ms_config)

# Prepare tuning for Postgres
print("Reading Postgres configuration file")
pg_config = configparser.ConfigParser()
with open('/home/ubuntu/liter2src/configs/pg_default.conf') as stream:
    pg_config.read_string("[dummysection]\n" + stream.read())
print("Creating Postgres evaluation object")
pg_eval = Evaluate.PostgresEval(pg_config)

# Select database system to tune
config = pg_config
db_eval = pg_eval

# Initialize file for result output
f = open("tpch.txt", "w")
f.write('param\tval\tquery\terror\tmillis')
f.flush()

# Iterate over parameter sections
for section in config.sections():
    # Iterate over parameters in section
    for param in config[section]:
        # Extract original parameter value
        print(f'Param: {param}', flush=True)
        val = config[section][param]
        print(f"Parameter {param} with value {val}", flush=True)
        # Iterate over alternative values
        alt_vals = alternative_vals(val)
        print(f"Alternative values: {alt_vals}", flush=True)
        if len(alt_vals) > 1:
            for alt_val in alt_vals:
                config[section][param] = alt_val
                print(f'Alternative {alt_val} to {val}', 
                      flush=True)
                error, times = db_eval.tpch_eval()
                for q in range(22):
                    f.write(f'{param}\t{alt_val}\t{q+1}' 
                            f'\t{error}\t{times[q]}\n')
                    f.flush()

# Close benchmark output file
f.close()