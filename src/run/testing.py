'''
Created on Apr 14, 2021

@author: immanueltrummer
'''
from doc.collection import DocCollection
from dbms.postgres import PgConfig
from dbms.mysql import MySQLconfig
from all.agents import VQN
from all.approximation import QNetwork
from all.environments import GymEnvironment
from all.experiments import run_experiment
from all.logging import DummyWriter
from all.policies.greedy import GreedyPolicy
from torch.optim import Adam
from torch import nn
from all.presets.classic_control import dqn
from environment.multi_doc import MultiDocTuning
from environment.supervised import LabeledDocTuning
from environment.hybrid import HybridDocTuning
from benchmark.evaluate import OLAP, TpcC
from search.objectives import Objective

# Initialize unsupervised hint extraction environment
dbms = MySQLconfig('tpch', 'root', 'mysql1234-', '/usr/local/mysql/bin')
#dbms = PgConfig(db='tpch', user='immanueltrummer')

success = dbms.set_param_smart('innodb_buffer_pool_size', '2GBMySQL')
print(success)

# print(dbms.query_one('show global variables'))
#
# c = dbms.connection.cursor(buffered=True)
# c.execute('show global variables where variable_name != \'keyring_file_data\'')
# var_vals = c.fetchall()
# for var_val in var_vals:
    # var, val = var_val
    # success = dbms.set_param(var, 'default')
    # print(f'Success {success} for variable {var}')
# c.close()

#
# unlabeled_docs = DocCollection('/Users/immanueltrummer/git/literateDBtuners/tuning_docs/postgres100', dbms)
# #unlabeled_docs = DocCollection('/Users/immanueltrummer/git/literateDBtuners/tuning_docs/mysql100', dbms)
# tpc_c = TpcC('/Users/immanueltrummer/benchmarks/oltpbench', 
            # '/Users/immanueltrummer/benchmarks/oltpbench/config/tpcc_config_postgres.xml', 
            # '/Users/immanueltrummer/benchmarks/oltpbench/results', dbms, 'tpcctemplate', 'tpcc')
            #
# metrics = tpc_c.evaluate()
# print(metrics)
