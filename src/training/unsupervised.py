'''
Created on May 5, 2021

@author: immanueltrummer

Train interpreting tuning documents without supervision.
'''
from all.environments.gym import GymEnvironment
from all.experiments import run_experiment
from all.presets.classic_control import dqn
from argparse import ArgumentParser
from benchmark.evaluate import OLAP
from dbms.postgres import PgConfig
from doc.collection import DocCollection
from environment.multi_doc import MultiDocTuning
from models.bert_tuning import BertFineTuning
from parameters.util import read_numerical
from search.objectives import Objective

# Parse command line arguments
parser = ArgumentParser(description='Unsupervised training for NLU of DB tuning documents')
parser.add_argument('device', type=str, help='"cuda" or "cpu"')
parser.add_argument('docs', type=str, help='Path to file with tuning documents')
parser.add_argument('queries', type=str, help='Path to file containing SQL queries')
parser.add_argument('logging', type=str, help='Path to file for benchmark logging')
parser.add_argument('user', type=str, help='User name for database access')
parser.add_argument('db', type=str, help='Name of database for tuning')
parser.add_argument('password', type=str, help='Password for database access')
parser.add_argument('parameters', type=str, help='Path to file with parameters')
parser.add_argument('data', type=str, help='Path to Postgres data directory')
parser.add_argument('restart', type=str, help='Command for restarting DBMS')
parser.add_argument('imodel', type=str, default='bert-base-cased',
                    help='Name or path to read initial model')
parser.add_argument('omodel', type=str, help='Path to write output model')
args = parser.parse_args()

device = args.device
path_to_docs = args.docs
path_to_queries = args.queries
log_path = args.logging
db_user = args.user
db_name = args.db
password = args.password
path_to_conf = args.parameters
path_to_data = args.data
restart_cmd = args.restart
input_model = args.imodel
output_model = args.omodel

# Initialize tuning documents
docs = DocCollection(docs_path=path_to_docs, 
                     dbms=None, size_threshold=0)

# Configure Postgres database system
pg_params = read_numerical(path_to_conf)
postgres = PgConfig(db=db_name, user=db_user, 
                    password=password,
                    restart_cmd=restart_cmd,
                    data_dir=path_to_data)
postgres.reset_config()
postgres.reconfigure()

# Initialize benchmark
bench = OLAP(postgres, path_to_queries)
bench.reset(log_path)

# Initialize environment
unsupervised_env = MultiDocTuning(
    docs=docs, dbms=postgres, benchmark=bench, 
    hardware=[2000000, 2000000, 8], hints_per_episode=1,
    nr_evals=1, scale_perf=0.01, objective=Objective.TIME)
unsupervised_env = GymEnvironment(unsupervised_env, device=device)

# Initialize agents
model = BertFineTuning(input_model)
def make_model(env):
    return model
agent = dqn(model_constructor=make_model, minibatch_size=2, device=device, 
            lr=1e-5, final_exploration_frame=10000, target_update_frequency=1, 
            replay_start_size=50)

# Run experiments
run_experiment(agents=agent, envs=unsupervised_env, frames=10000, test_episodes=10)

# Save final model
model.save_pretrained(output_model)