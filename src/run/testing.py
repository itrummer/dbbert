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
#dbms = MySQLconfig('tpch', 'root', 'mysql1234-')
dbms = PgConfig(db='tpch', user='immanueltrummer')

unlabeled_docs = DocCollection('/Users/immanueltrummer/git/literateDBtuners/tuning_docs/postgres100', dbms)
#unlabeled_docs = DocCollection('/Users/immanueltrummer/git/literateDBtuners/tuning_docs/mysql100', dbms)
tpc_c = TpcC('/Users/immanueltrummer/benchmarks/oltpbench', 
            '/Users/immanueltrummer/benchmarks/oltpbench/config/tpcc_config_postgres.xml', 
            '/Users/immanueltrummer/benchmarks/oltpbench/results', dbms, 'tpcctemplate', 'tpcc')

metrics = tpc_c.evaluate()
print(metrics)
