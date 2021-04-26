'''
Created on Apr 25, 2021

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

# Generate the agent
device = 'cpu'
writer = DummyWriter()
model = nn.Sequential(
        nn.Linear(1537, 128),
        # nn.ReLU(),
        # nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(128, 5),
    )
def make_model(env):
    return model
# create a Pytorch optimizer for the model
optimizer = Adam(model.parameters(), lr=0.01)
# create an Approximation of the Q-function
q = QNetwork(model, optimizer, writer=writer)
# create a Policy object derived from the Q-function
policy = GreedyPolicy(q, 5, epsilon=0.1)
# instantiate the agent
vqn = VQN(q, policy, discount_factor=0.99)

def run_dbms_benchmarks(labeled_path, unlabeled_path, dbms, 
                        dbms_label, tpcc_config, tpcc_reset_every):
    """ Run benchmark for specific database management system. 
    
    Args:
        labeled_path: path to labeled documents (for supervised training)
        unlabeled_path: path to unlabeled documents with tuning hints
        dbms: object representing database management system to tune
        dbms_label: DBMS label used for log file name
        tpcc_config: name of configuration file for TPC-C (DBMS-specific)
        tpcc_reset_every: reset database after so many evaluations for TPC-C
    """
    # Create supervised learning environment
    labeled_docs = DocCollection(docs_path=labeled_path, dbms=None, size_threshold=0)
    supervised_env = LabeledDocTuning(docs=labeled_docs, nr_hints=500, label_path=labeled_path)
    # Create benchmarks
    tpc_c = TpcC('/Users/immanueltrummer/benchmarks/oltpbench', 
                f'/Users/immanueltrummer/benchmarks/oltpbench/config/{tpcc_config}', 
                '/Users/immanueltrummer/benchmarks/oltpbench/results', 
                dbms, 'tpccsf20', 'tpcc', tpcc_reset_every)
    tpc_h = OLAP(dbms, 
                 '/Users/immanueltrummer/git/literateDBtuners/benchmarking/tpch/queries.sql')
    # Iterate over benchmarks
    tpc_c_descr = (tpc_c, 'TPCC', Objective.THROUGHPUT)
    tpc_h_descr = (tpc_h, 'TPCH', Objective.TIME)
    for bench, bench_label, objective in [tpc_c_descr, tpc_h_descr]:
        # Create unsupervised learning environment
        unlabeled_docs = DocCollection(unlabeled_path, dbms)
        unsupervised_env = MultiDocTuning(docs=unlabeled_docs, dbms=dbms, benchmark=bench, 
                                          hardware=[2000000, 2000000, 8], nr_hints=100, 
                                          nr_rereads=50, nr_evals=2, objective)
        # Iterate over supervision style
        for sup_episodes, sup_label in [(0, 'nosup'), (100, 'sup')]:
            # Create hybrid (unsupervised+supervised) environment
            hybrid_env = HybridDocTuning(supervised_env, unsupervised_env, sup_episodes)
            hybrid_env = GymEnvironment(hybrid_env)
            # Initialize logging and run experiment
            log_path = f'/Users/immanueltrummer/git/literateDBtuners/bench_results/{dbms_label}_{bench_label}_{sup_label}'
            bench.reset(log_path)
            run_experiment(agents=dqn(model_constructor=make_model), 
                           envs=hybrid_env, frames=50000, test_episodes=0)

# Run experiments for Postgres
postgres = PgConfig(db='tpch', user='immanueltrummer')
run_dbms_benchmarks('/Users/immanueltrummer/git/literateDBtuners/tuning_docs/ms10docsLabels.csv', 
                    '/Users/immanueltrummer/git/literateDBtuners/tuning_docs/postgres100',
                    postgres, 'pg', 'tpcc_config_postgres.xml', 1)

# Run experiments for MySQL
mysql = MySQLconfig('tpch', 'root', 'mysql1234-', '/usr/local/mysql/bin')
run_dbms_benchmarks('/Users/immanueltrummer/git/literateDBtuners/tuning_docs/pg10docsLabels.csv', 
                    '/Users/immanueltrummer/git/literateDBtuners/tuning_docs/mysql100',
                    mysql, 'ms', 'tpcc_config_mysql.xml', 10)