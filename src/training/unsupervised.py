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
from configparser import ConfigParser
from dbms.mysql import MySQLconfig
from dbms.postgres import PgConfig
from doc.collection import DocCollection
from environment.multi_doc import MultiDocTuning
from models.bert_tuning import BertFineTuning
from parameters.util import read_numerical
import benchmark.factory
import dbms.factory
import search.objectives

# Read configuration referenced as command line parameters
arg_parser = ArgumentParser(description='DB-BERT: Train NLU for DB tuning documents')
arg_parser.add_argument('cpath', type=str, help='Path to configuration file')
args = arg_parser.parse_args()
config = ConfigParser()
config.read(args.cpath)

device = config['LEARNING']['device'] # cuda or cpu
input_model = config['LEARNING']['input'] # name or path to input model
output_model = config['LEARNING']['output'] # path to output model
nr_frames = int(config['LEARNING']['nr_frames']) # number of frames
epsilon = float(config['LEARNING']['start_epsilon']) # start value for epsilon
p_scaling = float(config['LEARNING']['performance_scaling']) # scaling for performance reward
a_scaling = float(config['LEARNING']['assignment_scaling']) # assignment reward scaling
nr_evals = int(config['LEARNING']['nr_evaluations']) # number of evaluations per episode
nr_hints = int(config['LEARNING']['nr_hints']) # number of hints per episode
min_batch_size = int(config['LEARNING']['min_batch_size']) # samples per batch
mask_params = True if config['LEARNING']['mode'] == 'masked' else False

dbms_name = config['DATABASE']['dbms']
db_user = config['DATABASE']['user']
db_name = config['DATABASE']['name']
password = config['DATABASE']['password']
restart_cmd = config['DATABASE']['restart_cmd']
bin_dir = config['DATABASE']['bin_dir']
path_to_data = config['DATABASE']['data_dir']
path_to_conf = config['DATABASE']['config']

path_to_docs = config['BENCHMARK']['docs']
max_length = int(config['BENCHMARK']['max_length'])
path_to_queries = config['BENCHMARK']['queries']
log_path = config['BENCHMARK']['logging']
memory = float(config['BENCHMARK']['memory'])
disk = float(config['BENCHMARK']['disk'])
cores = float(config['BENCHMARK']['cores'])

objective = search.objectives.from_file(config)
dbms = dbms.factory.from_file(config)
bench = benchmark.factory.from_file(config, dbms)
docs = DocCollection(docs_path=path_to_docs, 
    dbms=None, size_threshold=max_length)

dbms.reset_config()
dbms.reconfigure()
bench.reset(log_path)

# Initialize environment
unsupervised_env = MultiDocTuning(
    docs=docs, max_length=max_length, mask_params=mask_params,
    dbms=dbms, benchmark=bench, hardware=[memory, disk, memory],
    hints_per_episode=nr_hints, nr_evals=nr_evals,
    scale_perf=p_scaling, scale_asg=a_scaling, objective=objective)
unsupervised_env = GymEnvironment(unsupervised_env, device=device)

# Initialize agents
model = BertFineTuning(input_model)
agent = dqn(
    model_constructor=lambda _:model, minibatch_size=min_batch_size, 
    device=device, lr=1e-5, initial_exploration=epsilon, replay_start_size=50, 
    final_exploration_frame=nr_frames, target_update_frequency=1)

# Run experiments
run_experiment(agents=agent, envs=unsupervised_env, 
               frames=nr_frames, test_episodes=1)

# Save final model
model.model.save_pretrained(output_model)