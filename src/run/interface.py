'''
Created on Jan 7, 2022

@author: immanueltrummer
'''
import streamlit as st
import sys

sys.path.append('/Users/immanueltrummer/git/literateDBtuners/src/')

from pybullet_utils.util import set_global_seeds
from configparser import ConfigParser
from doc.collection import DocCollection
from environment.zero_shot import NlpTuningEnv
from stable_baselines3 import DQN
from stable_baselines3 import A2C
from stable_baselines3.common.utils import set_random_seed
import benchmark.factory
import dbms.factory
import environment.multi_doc
import numpy as np
import random
import search.objectives
import time
import torch
import pandas as pd

st.set_page_config(page_title='DB-BERT', layout='wide')
st.header('DB-BERT Demonstration')
st.markdown('DB-BERT uses hints mined from text for database tuning.')

config_dir = '/Users/immanueltrummer/git/literateDBtuners/config/'
default_config = ConfigParser()
default_config.read(config_dir + 'Defaults')

device = default_config['LEARNING']['device'] # cuda or cpu
nr_frames = int(default_config['LEARNING']['nr_frames']) # number of frames
timeout_s = float(default_config['LEARNING']['timeout_s']) # seconds until timeout
p_scaling = float(default_config['LEARNING']['performance_scaling']) # scaling for performance reward
a_scaling = float(default_config['LEARNING']['assignment_scaling']) # assignment reward scaling
nr_evals = int(default_config['LEARNING']['nr_evaluations']) # number of evaluations per episode
nr_hints = int(default_config['LEARNING']['nr_hints']) # number of hints per episode
min_batch_size = int(default_config['LEARNING']['min_batch_size']) # samples per batch

nr_runs = int(default_config['BENCHMARK']['nr_runs'])
# path_to_docs = default_config['BENCHMARK']['docs']
max_length = int(default_config['BENCHMARK']['max_length'])
filter_params = int(default_config['BENCHMARK']['filter_param'])
use_implicit = int(default_config['BENCHMARK']['use_implicit'])
hint_order = environment.multi_doc.parse_order(default_config)
log_path = default_config['BENCHMARK']['logging']
memory = float(default_config['BENCHMARK']['memory'])
disk = float(default_config['BENCHMARK']['disk'])
cores = float(default_config['BENCHMARK']['cores'])

dbms_label = st.selectbox('Select DBMS: ', ['Postgres', 'MySQL'], index=0)
bench_label = st.selectbox('Select Benchmark: ', ['TPC-H', 'TPC-C'], index=0)
obj_label = st.selectbox('Select Metric: ', ['Latency', 'Throughput'], index=0)
path_to_docs = st.text_input(
    'Enter Path to Text: ', 
    '/Users/immanueltrummer/git/literateDBtuners/tuning_docs/pg_tpch_single')

nr_frames = st.number_input('Enter Iteration Limit: ', min_value=1, max_value=500, value=1)
timeout_s = st.number_input('Enter Timeout (s): ', min_value=60, max_value=1500, value=600)

obj_config = ConfigParser()
obj_config.read(config_dir + obj_label)
dbms_config = ConfigParser()
dbms_config.read(config_dir + dbms_label)
bench_config = ConfigParser()
bench_config.read(config_dir + bench_label)

objective = search.objectives.from_file(obj_config)
dbms = dbms.factory.from_file(dbms_config)
bench = benchmark.factory.from_file(bench_config, dbms)

if st.button('Start Tuning'):
    st.write('Starting tuning session ...')
    # Initialize for new run
    dbms.reset_config()
    dbms.reconfigure()
    bench.reset(log_path, 0)
    
    # Initialize input documents
    st.write(f'Pre-processing input text at "{path_to_docs}" ...')
    docs = DocCollection(
        docs_path=path_to_docs, dbms=dbms, size_threshold=max_length,
        use_implicit=use_implicit, filter_params=filter_params)
    st.write('Pre-processing of input text is finished.')
    
    st.write('Extracted Hints: ')
    hints = docs.get_hints(0)
    params = [h.param.group() for h in hints]
    values = [h.value.group() for h in hints]
    passages = [h.passage for h in hints]
    df = pd.DataFrame({'Parameter':params, 'Value':values, 'Text Passage':passages})
    st.table(df)
    # for hint in docs.get_hints(0):
        # st.write(hint)
    
    # Initialize environment
    set_random_seed(0)
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(0)
    set_global_seeds(0)
    
    unsupervised_env = NlpTuningEnv(
        docs=docs, max_length=max_length, hint_order=hint_order, 
        dbms=dbms, benchmark=bench, hardware=[memory, disk, cores], 
        hints_per_episode=nr_hints, nr_evals=nr_evals, 
        scale_perf=p_scaling, scale_asg=a_scaling, objective=objective)
    unsupervised_env.reset()
    # unsupervised_env = GymEnvironment(unsupervised_env, device=device)
    
    # Initialize agents
    model = A2C(
        'MlpPolicy', unsupervised_env, 
        verbose=1, normalize_advantage=True)
        
    # Warm-up phase (quick), followed by actual tuning
    st.write(f'Running for up to {timeout_s} seconds, {nr_frames} frames')
    start_s = time.time()
    # Could move warmup to pre-training
    model.learn(total_timesteps=20000)
    unsupervised_env.stop_warmup()
    for i in range(nr_frames):
        model.learn(total_timesteps=1)
        elapsed_s = time.time() - start_s
        if elapsed_s > timeout_s:
            break
        if i % 500 == 0:
            st.write(f'Step {i} - tuned for {elapsed_s} seconds')
            
    # Show final summary
    st.write('Tuning process of DB-BERT is finished.')
    # st.write('Summary of results:')
    bench.print_stats()