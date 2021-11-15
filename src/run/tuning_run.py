'''
Created on Nov 13, 2021

@author: immanueltrummer
'''
from pybullet_utils.util import set_global_seeds
'''
Created on May 5, 2021

@author: immanueltrummer

Train interpreting tuning documents without supervision.
'''
from argparse import ArgumentParser
from configparser import ConfigParser
from doc.collection import DocCollection
from environment.multi_doc import MultiDocBart
from stable_baselines3 import DQN
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
import benchmark.factory
import dbms.factory
import environment.multi_doc
import numpy as np
import random
import search.objectives
import time
import torch

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
timeout_s = float(config['LEARNING']['timeout_s']) # seconds until timeout
epsilon = float(config['LEARNING']['start_epsilon']) # start value for epsilon
p_scaling = float(config['LEARNING']['performance_scaling']) # scaling for performance reward
a_scaling = float(config['LEARNING']['assignment_scaling']) # assignment reward scaling
nr_evals = int(config['LEARNING']['nr_evaluations']) # number of evaluations per episode
nr_hints = int(config['LEARNING']['nr_hints']) # number of hints per episode
min_batch_size = int(config['LEARNING']['min_batch_size']) # samples per batch
mask_params = True if config['LEARNING']['mode'] == 'masked' else False

nr_runs = int(config['BENCHMARK']['nr_runs'])
path_to_docs = config['BENCHMARK']['docs']
use_recs = int(config['BENCHMARK']['use_recs'])
if 'rec_file' in config['BENCHMARK']:
    rec_path = config['BENCHMARK']['rec_file']
else:
    rec_path = None
max_length = int(config['BENCHMARK']['max_length'])
filter_params = int(config['BENCHMARK']['filter_param'])
use_implicit = int(config['BENCHMARK']['use_implicit'])
hint_order = environment.multi_doc.parse_order(config)
log_path = config['BENCHMARK']['logging']
memory = float(config['BENCHMARK']['memory'])
disk = float(config['BENCHMARK']['disk'])
cores = float(config['BENCHMARK']['cores'])

objective = search.objectives.from_file(config)
dbms = dbms.factory.from_file(config)
bench = benchmark.factory.from_file(config, dbms)

for run_ctr in range(nr_runs):
    print(f'Starting run number {run_ctr} ...')
    
    # Initialize for new run
    dbms.reset_config()
    dbms.reconfigure()
    bench.reset(log_path, run_ctr)
    
    # Initialize input documents
    print(f'Pre-processing input text at "{path_to_docs}" ...')
    docs = DocCollection(
        docs_path=path_to_docs, dbms=dbms, size_threshold=max_length,
        use_implicit=use_implicit, filter_params=filter_params)
    print('Pre-processing of input text is finished.')
    
    # Initialize environment
    set_random_seed(0)
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(0)
    set_global_seeds(0)

    unsupervised_env = MultiDocBart(
        docs=docs, max_length=max_length, mask_params=mask_params, 
        hint_order=hint_order, dbms=dbms, benchmark=bench, 
        hardware=[memory, disk, cores], hints_per_episode=nr_hints, 
        nr_evals=nr_evals, scale_perf=p_scaling, scale_asg=a_scaling, 
        objective=objective, rec_path=rec_path, use_recs=use_recs)
    unsupervised_env.reset()
    # unsupervised_env = GymEnvironment(unsupervised_env, device=device)
    
    # Initialize agents
    model = A2C('MlpPolicy', unsupervised_env, verbose=1, seed=0)
    
    # Warm-up phase (quick), followed by actual tuning
    print(f'Running for up to {timeout_s} seconds, {nr_frames} frames')
    start_s = time.time()
    model.learn(total_timesteps=10000)
    unsupervised_env.stop_warmup()
    for i in range(nr_frames):
        model.learn(total_timesteps=1)
        elapsed_s = time.time() - start_s
        if elapsed_s > timeout_s:
            break
        if i % 500 == 0:
            print(f'Step {i} - tuned for {elapsed_s} seconds')
    
    # Show final summary
    print('Tuning process of DB-BERT is finished.')
    print('Summary of results:')
    bench.print_stats()