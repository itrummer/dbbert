'''
Created on Nov 11, 2020

@author: immanueltrummer
'''
""" Analyze impact of tuning parameters. """

import Configurations
import Evaluate

# Prepare tuning and perform NLP analysis
"""
print("About to create Postgres tuning config")
pg_config = Configurations.TuningConfig(
    '/etc/postgresql/10/main/postgresql.conf')
"""
print("Creating MySQL tuning and benchmarking objects")
mysql_config = Configurations.TuningConfig('/etc/mysql/my.cnf')
tpch_eval = Evaluate.MySQLeval(mysql_config)

# Iterate over all parameters and try different settings
f = open("mysqlTpchAnalysis.txt", "w")
for lineID, param in mysql_config.idToTunable.items():
    for factor in (0.2, 5, 1):
        mysql_config.set_scale(lineID, factor)
        print(f'Trying with factor {factor} for {param.name}')
        error, millis = tpch_eval.tpch_eval()
        print(f'Trying with factor {factor}')
        f.write(f'{param.name}\t{factor}\t{error}\t{millis}\n')
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