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
from baselines.ddpg import run_ddpg
from parameters.util import read_numerical, is_numerical

# This timeout (in seconds) is used by all relevant RL algorithms
LITERATE_TIMEOUT_S = 300

# Generate the agent
device = 'cpu'
writer = DummyWriter()
model = nn.Sequential(
        nn.Linear(1538, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 5),
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
                        dbms_label, tpcc_config, tpcc_reset_every,
                        all_params):
    """ Run benchmark for specific database management system. 
    
    Args:
        labeled_path: path to labeled documents (for supervised training)
        unlabeled_path: path to unlabeled documents with tuning hints
        dbms: object representing database management system to tune
        dbms_label: DBMS label used for log file name
        tpcc_config: name of configuration file for TPC-C (DBMS-specific)
        tpcc_reset_every: reset database after so many evaluations for TPC-C
        all_params: list of all tuning parameters for current DBMS
    """
    log_dir = f'/Users/immanueltrummer/git/literateDBtuners/bench_results'
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
        unsupervised_env = MultiDocTuning(
            docs=unlabeled_docs, dbms=dbms, benchmark=bench, 
            hardware=[2000000, 2000000, 8], nr_hints=100, 
            nr_rereads=50, nr_evals=2, objective=objective)
        # Iterate over supervision style
        for sup_episodes, sup_label in [(25, 'sup'), (0, 'nosup')]:
            # Reset to default configuration
            dbms.reset_config()
            dbms.reconfigure()
            # Create hybrid (unsupervised+supervised) environment
            hybrid_env = HybridDocTuning(supervised_env, unsupervised_env, sup_episodes)
            hybrid_env = GymEnvironment(hybrid_env)
            # Initialize logging and run experiment
            log_path = f'{log_dir}/main_{dbms_label}_{bench_label}_{sup_label}'
            print(f'Logging to {log_path} ...')
            bench.reset(log_path)
            run_experiment(agents=dqn(model_constructor=make_model), 
                           envs=hybrid_env, frames=50000, test_episodes=0,
                           timeout_s=LITERATE_TIMEOUT_S)
        # Run DDPG++ baseline
        for change_factor in [2, 10, 100]:
            # Reset to default configuration
            dbms.reset_config()
            dbms.reconfigure()
            # Initialize logging
            log_path = f'{log_dir}/ddpg_{change_factor}'
            print(f'Logging to {log_path} ...')
            bench.reset(log_path)
            run_ddpg(dbms, bench, objective, all_params, 
                     change_factor, LITERATE_TIMEOUT_S)

# Run experiments for MySQL
mysql = MySQLconfig('tpch', 'root', 'mysql1234-', '/usr/local/mysql/bin')
ms_p_vals = mysql.all_params()
ms_params = [p for p, v in ms_p_vals if is_numerical(v)]
run_dbms_benchmarks('/Users/immanueltrummer/git/literateDBtuners/tuning_docs/pg10docsLabels.csv', 
                    '/Users/immanueltrummer/git/literateDBtuners/tuning_docs/mysql100',
                    mysql, 'ms', 'tpcc_config_mysql.xml', 1000000, ms_params)

# Run experiments for Postgres
pg_params = read_numerical('/Users/immanueltrummer/git/literateDBtuners/config/pg13.conf')
postgres = PgConfig(db='tpch', user='immanueltrummer')
run_dbms_benchmarks('/Users/immanueltrummer/git/literateDBtuners/tuning_docs/ms10docsLabels.csv', 
                    '/Users/immanueltrummer/git/literateDBtuners/tuning_docs/postgres100',
                    postgres, 'pg', 'tpcc_config_postgres.xml', 1, pg_params)