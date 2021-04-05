'''
Created on Nov 11, 2020

@author: immanueltrummer
'''
""" Analyze impact of tuning parameters. """

import configparser
from evaluation import Evaluate
import re
import sys

# Extract command line parameters
db_system = sys.argv[1]
benchmark = sys.argv[2]
out_path = sys.argv[3]

# Distinguish database system
db_config = None
db_eval = None
if db_system == 'ms':
    # Prepare tuning for MySQL
    print("Reading default MySQL configuration file")
    config = configparser.ConfigParser()
    config.read('mysql/all_tunable_config.cnf')
    print("Creating MySQL evaluation object")
    db_eval = Evaluate.MySQLeval(config)
elif db_system == 'pg':
    # Prepare tuning for Postgres
    print("Reading Postgres configuration file")
    config = configparser.ConfigParser()
    with open('/home/ubuntu/liter2src/configs/pg_default.conf') as stream:
        config.read_string("[dummysection]\n" + stream.read())
    print("Creating Postgres evaluation object")
    db_eval = Evaluate.PostgresEval(config)
else:
    print(f'Unsupported database system: {db_system}!')
    sys.exit(1)

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

def bench(benchmark, out_file):
    """ Run benchmark and write result to file """
    if benchmark == 'tpcc':
        error, throughput = db_eval.tpcc_eval()
        out_file.write(f'tpcc\t{error}\t{throughput}\n')
    elif benchmark == 'tpch':
        # Select TPC-H queries to tune
        tpch_queries = [2]
        error, times = db_eval.tpch_eval(tpch_queries)
        for q in tpch_queries:
            out_file.write(f'{q+1}\t{error}\t{times[q]}\n')
    else:
        print(f'Unknown benchmark: {benchmark}!')
        sys.exit(1)

# Open benchmark result file
with open(out_path, 'w') as f:
    # Write benchmark file header
    f.write('param\tval\tquery\terror\tmetric\n')
    f.flush()
    # Iterate over parameter sections
    for section in config.sections():
        # Iterate over parameters in section
        for param in config[section]:
            # Extract original parameter value
            print(f'Param: {param}', flush=True)
            val = config[section][param]
            print(f"Parameter {param} with value {val}", 
                  flush=True)
            # Iterate over alternative values
            alt_vals = alternative_vals(val)
            print(f"Alternative values: {alt_vals}", 
                  flush=True)
            if len(alt_vals) > 1:
                for alt_val in alt_vals:
                    print(f'Alt. {alt_val} to {val}', flush=True)
                    config[section][param] = alt_val
                    alt_val_clean = alt_val.replace('\t', '')
                    f.write(f'{param}\t{alt_val_clean}\t')
                    bench(benchmark, f)
                    f.flush()