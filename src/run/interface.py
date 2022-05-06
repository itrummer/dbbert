'''
Created on Jan 7, 2022

@author: immanueltrummer
'''
import os
import pathlib
import streamlit as st
import sys

cur_file_dir = os.path.dirname(__file__)
src_dir = pathlib.Path(cur_file_dir).parent
sys.path.append(str(src_dir))
print(sys.path)

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

def get_value(configuration, category, property_, default):
    """ Get value from configuration or defaults.
    
    Args:
        configuration: contains values for configuration properties
        category: category of configuration property
        property_: retrieve value for this property
        default: default value if property not stored
    
    Returns:
        value for given property
    """
    if not category in configuration:
        return default
    else:
        return configuration[category].get(property_, default)


st.set_page_config(page_title='DB-BERT', layout='wide')
st.header('DB-BERT Demonstration')
st.markdown('DB-BERT uses hints mined from text for database tuning.')

root_dir = src_dir.parent
config_dir = root_dir.joinpath('config')
config = ConfigParser()
config.read(str(config_dir.joinpath('Defaults')))

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with st.expander('Text Analysis'):
    def_max_length = int(get_value(config, 'BENCHMARK', 'max_length', 128))
    def_batch_size = int(get_value(config, 'BENCHMARK', 'min_batch_size', 8))
    def_filter_params = int(get_value(config, 'BENCHMARK', 'filter_param', 1))
    def_use_implicit = int(get_value(config, 'BENCHMARK', 'use_implicit', 1))
    def_order_id = int(get_value(config, 'BENCHMARK', 'hint_order', 2))
    max_length = st.number_input(
        'Characters per Text Block', value=def_max_length)
    min_batch_size = st.number_input('Text Batch Size', value=def_batch_size)
    filter_params = st.selectbox(
        'Heuristic Text Filter', index=def_filter_params, 
        options=range(2), format_func=lambda i:['No', 'Yes'][i])
    use_implicit = st.selectbox(
        'Implicit Parameter References', index=def_use_implicit, 
        options=range(2), format_func=lambda i:['No', 'Yes'][i])
    hint_order_id = st.selectbox(
        'Order Hints', index=def_order_id, options=range(3),
        format_func=lambda i:[
            'Document Order', 
            'Frequent Parameters First', 
            'Frequent Parameters First with Limit'][i])
    hint_order = [
        environment.multi_doc.HintOrder.DOCUMENT, 
        environment.multi_doc.HintOrder.BY_PARAMETER, 
        environment.multi_doc.HintOrder.BY_STRIDE][hint_order_id]

with st.expander('Reinforcement Learning'):
    def_nr_frames = int(get_value(config, 'LEARNING', 'nr_frames', 1))
    def_timeout_s = int(get_value(config, 'LEARNING', 'timeout_s', 120))
    def_p_scaling = float(get_value(
        config, 'LEARNING', 'performance_scaling', 0.1))
    def_a_scaling = float(get_value(
        config, 'LEARNING', 'assignment_scaling', 0.1))
    def_nr_evals = int(get_value(config, 'LEARNING', 'nr_evaluations', 1))
    def_nr_hints = int(get_value(config, 'LEARNING', 'nr_hints', 1))
    nr_frames = st.number_input('Number of Frames', value=def_nr_frames)
    timeout_s = st.number_input('Timeout in Seconds', value=def_timeout_s)
    p_scaling = st.number_input(
        'Reward Weight for Performance', value=def_p_scaling)
    a_scaling = st.number_input(
        'Reward Weight for Successful Setting', value=def_a_scaling)
    nr_evals = st.number_input(
        'Number of Configurations Evaluated per Episode', value=def_nr_evals)
    nr_hints = st.number_input(
        'Number of Hints Processed per Episode', value=def_nr_hints)

with st.expander('Hardware Properties'):
    def_mem = float(get_value(config, 'BENCHMARK', 'memory', 8000000))
    def_disk = float(get_value(config, 'BENCHMARK', 'disk', 500000000))
    def_cores = int(get_value(config, 'BENCHMARK', 'cores', 8))
    memory = st.number_input('Main Memory (bytes)', value=def_mem)
    disk = st.number_input('Disk Space (bytes)', value=def_disk)
    cores = st.number_input('Number of Cores', value=def_cores)

#device = config['LEARNING']['device'] # cuda or cpu
nr_frames = int(config['LEARNING']['nr_frames']) # number of frames
timeout_s = float(config['LEARNING']['timeout_s']) # seconds until timeout
p_scaling = float(config['LEARNING']['performance_scaling']) # scaling for performance reward
a_scaling = float(config['LEARNING']['assignment_scaling']) # assignment reward scaling
nr_evals = int(config['LEARNING']['nr_evaluations']) # number of evaluations per episode
nr_hints = int(config['LEARNING']['nr_hints']) # number of hints per episode
min_batch_size = int(config['LEARNING']['min_batch_size']) # samples per batch

nr_runs = int(config['BENCHMARK']['nr_runs'])
# path_to_docs = config['BENCHMARK']['docs']
log_path = config['BENCHMARK']['logging']

dbms_label = st.selectbox('Select DBMS: ', ['Postgres', 'MySQL'], index=0)
bench_label = st.selectbox('Select Benchmark: ', ['TPC-H', 'TPC-C'], index=0)
obj_label = st.selectbox('Select Metric: ', ['Latency', 'Throughput'], index=0)
path_to_docs = st.text_input(
    'Enter Path to Text: ', 
    '/Users/immanueltrummer/git/literateDBtuners/tuning_docs/pg_tpch_single')

nr_frames = st.number_input('Enter Iteration Limit: ', min_value=1, max_value=500, value=1)
timeout_s = st.number_input('Enter Timeout (s): ', min_value=60, max_value=1500, value=600)


if st.button('Start Tuning'):
    
    obj_config = ConfigParser()
    obj_config.read(str(config_dir.joinpath(obj_label)))
    dbms_config = ConfigParser()
    dbms_config.read(str(config_dir.joinpath(dbms_label)))
    bench_config = ConfigParser()
    bench_config.read(str(config_dir.joinpath(bench_label)))
    
    objective = search.objectives.from_file(obj_config)
    dbms = dbms.factory.from_file(dbms_config)
    bench = benchmark.factory.from_file(bench_config, dbms)
    
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