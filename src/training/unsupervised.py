'''
Created on May 5, 2021

@author: immanueltrummer

Train interpreting tuning documents without supervision.
'''
from all.environments.gym import GymEnvironment
from all.experiments import run_experiment
from all.presets.classic_control import dqn
from benchmark.evaluate import OLAP
from dbms.postgres import PgConfig
from environment.multi_doc import MultiDocTuning
from models.bert_tuning import BertFineTuning
from parameters.util import read_numerical
from search.objectives import Objective

device = 'cuda'
path_to_docs = ''
path_to_queries = ''
log_path = ''
db_user = 'postgres'
db_name = 'tpch'
path_to_conf = ''

# Configure Postgres database system
pg_params = read_numerical(path_to_conf)
postgres = PgConfig(db=db_name, user=db_user)
postgres.reset_config()
postgres.reconfigure()

# Initialize benchmark
bench = OLAP(postgres, path_to_queries)
bench.reset(log_path)

# Initialize environment
unsupervised_env = MultiDocTuning(
    docs=path_to_docs, dbms=postgres, benchmark=bench, 
    hardware=[2000000, 2000000, 8], hints_per_episode=1,
    nr_evals=1, objective=Objective.TIME)
unsupervised_env = GymEnvironment(unsupervised_env)

# Initialize agents
model = BertFineTuning()
def make_model(env):
    return model
agent = dqn(model_constructor=make_model, minibatch_size=2, device=device, 
            lr=1e-5, final_exploration_frame=1000, target_update_frequency=1, 
            replay_start_size=50)

# Run experiments
run_experiment(agents=agent, envs=unsupervised_env, frames=1000, test_episodes=10)