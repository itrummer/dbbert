'''
Created on May 5, 2021

@author: immanueltrummer

Train interpreting tuning documents without supervision.
'''
from all.environments.gym import GymEnvironment
from all.experiments.single_env_experiment import SingleEnvExperiment
from all.presets.classic_control import dqn, ddqn
from argparse import ArgumentParser
from configparser import ConfigParser
from doc.collection import DocCollection
from environment.multi_doc import MultiDocTuning
from models.bert_tuning import BertFineTuning
import benchmark.factory
import dbms.factory
import environment.multi_doc
import numpy as np
import search.objectives
import time

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
    
    print(f'Starting run number {run_ctr}')
    
    # Initialize for new run
    dbms.reset_config()
    dbms.reconfigure()
    bench.reset(log_path, run_ctr)
    
    docs = DocCollection(
        docs_path=path_to_docs, dbms=dbms, size_threshold=max_length,
        use_implicit=use_implicit, filter_params=filter_params)
    
    # Initialize environment
    unsupervised_env = MultiDocTuning(
        docs=docs, max_length=max_length, mask_params=mask_params, 
        hint_order=hint_order, dbms=dbms, benchmark=bench, 
        hardware=[memory, disk, cores], hints_per_episode=nr_hints, 
        nr_evals=nr_evals, scale_perf=p_scaling, scale_asg=a_scaling, 
        objective=objective, rec_path=rec_path, use_recs=use_recs)
    unsupervised_env = GymEnvironment(unsupervised_env, device=device)
    
    # Initialize agents
    model = BertFineTuning(input_model)
    agent = ddqn(
        model_constructor=lambda _:model, minibatch_size=min_batch_size, 
        device=device, lr=1e-5, initial_exploration=epsilon, replay_start_size=50, 
        final_exploration_frame=nr_frames, target_update_frequency=1)
    # agent = dqn(
        # model_constructor=lambda _:model, minibatch_size=min_batch_size, 
        # device=device, lr=1e-5, initial_exploration=epsilon, replay_start_size=50, 
        # final_exploration_frame=nr_frames, target_update_frequency=1)
    
    # Run experiments
    experiment = SingleEnvExperiment(
        agent, unsupervised_env, logdir='runs', 
        quiet=False, render=False, write_loss=True)
    
    def finished(experiment, elapsed_s):
        """ Returns true iff the experiment is finished. """
        return elapsed_s > timeout_s or experiment._done(
            frames=nr_frames, episodes=np.inf)
    
    print(f'Running for up to {timeout_s} seconds, {nr_frames} frames')
    start_s = time.time()
    elapsed_s = 0
    while not finished(experiment, elapsed_s):
        experiment._run_training_episode()
        cur_s = time.time()
        elapsed_s = cur_s - start_s
        print(f'Elapsed time: {elapsed_s} seconds')
    
    # Save final model
    model.model.save_pretrained(output_model)