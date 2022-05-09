'''
Created on Jan 7, 2022

@author: immanueltrummer
'''
import altair as alt
import os
import pathlib
import streamlit as st
import sys

cur_file_dir = os.path.dirname(__file__)
src_dir = pathlib.Path(cur_file_dir).parent
root_dir = src_dir.parent
sys.path.append(str(src_dir))
sys.path.append(str(root_dir))
print(sys.path)

from dbms.postgres import PgConfig
from dbms.mysql import MySQLconfig
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

config_dir = root_dir.joinpath('demo_configs')
config_files = [f for f in config_dir.iterdir() if f.is_file()]
nr_configs = len(config_files)
config_idx = st.selectbox(
    'Default Configuration', options=range(nr_configs), 
    format_func=lambda i:str(config_files[i]), index=0)
config_path = str(config_files[config_idx])
config = ConfigParser()
config.read(config_path)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
log_path = 'log_db_bert'

text_dir = root_dir.joinpath('demo_docs')
text_files = [f for f in text_dir.iterdir() if f .is_file()]
nr_texts = len(text_files)
text_idx = st.selectbox(
    'Text Documents with Tuning Hints', options=range(nr_texts), 
    index=0, format_func=lambda i:str(text_files[i]))
path_to_docs = str(text_files[text_idx])

with st.expander('Text Analysis'):
    def_max_length = int(get_value(config, 'BENCHMARK', 'max_length', 128))
    def_batch_size = int(get_value(config, 'BENCHMARK', 'min_batch_size', 8))
    def_filter_params = int(get_value(config, 'BENCHMARK', 'filter_param', 1))
    def_use_implicit = int(get_value(config, 'BENCHMARK', 'use_implicit', 1))
    def_order_id = int(get_value(config, 'BENCHMARK', 'hint_order', 2))
    max_length = int(st.number_input(
        'Characters per Text Block', value=def_max_length))
    min_batch_size = int(st.number_input('Text Batch Size', value=def_batch_size))
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
    nr_frames = int(st.number_input('Number of Frames', value=def_nr_frames))
    timeout_s = int(st.number_input('Timeout in Seconds', value=def_timeout_s))
    p_scaling = st.number_input(
        'Reward Weight for Performance', value=def_p_scaling)
    a_scaling = st.number_input(
        'Reward Weight for Successful Setting', value=def_a_scaling)
    nr_evals = int(st.number_input(
        'Number of Configurations Evaluated per Episode', value=def_nr_evals))
    nr_hints = int(st.number_input(
        'Number of Hints Processed per Episode', value=def_nr_hints))

with st.expander('Hardware Properties'):
    def_mem = float(get_value(config, 'BENCHMARK', 'memory', 8000000))
    def_disk = float(get_value(config, 'BENCHMARK', 'disk', 500000000))
    def_cores = int(get_value(config, 'BENCHMARK', 'cores', 8))
    memory = st.number_input('Main Memory (bytes)', value=def_mem)
    disk = st.number_input('Disk Space (bytes)', value=def_disk)
    cores = st.number_input('Number of Cores', value=def_cores)

with st.expander('Database'):
    def_db = get_value(config, 'DATABASE', 'dbms', 'pg')
    def_db_name = get_value(config, 'DATABASE', 'name', '')
    def_db_user = get_value(config, 'DATABASE', 'user', 'ubuntu')
    def_db_pwd = get_value(config, 'DATABASE', 'password', '')
    def_restart = get_value(config, 'DATABASE', 'restart_cmd', '')
    def_recover = get_value(config, 'DATABASE', 'recovery_cmd', '')
    if def_db == 'ms':
        def_db_idx = 1
    else:
        def_db_idx = 0
    dbms_id = st.selectbox(
        'DBMS', options=range(2), index=def_db_idx,
        format_func=lambda i:['Postgres', 'MySQL'][i])
    db_name = st.text_input('Database Name', value=def_db_name)
    db_user = st.text_input('Database User', value=def_db_user)
    db_pwd = st.text_input(
        'Database Password', value=def_db_pwd, type='password')
    restart_cmd = st.text_input(
        'Command for DBMS Restart', value=def_restart)
    recover_cmd = st.text_input(
        'Command for DBMS Recovery', value=def_recover)

with st.expander('Benchmark'):
    benchmark_type = st.selectbox(
        'Benchmark Type', options=range(2), 
        format_func=lambda i:[
            'Minimize Run Time', 'Maximize Throughput'][i], index=0)
    if benchmark_type == 0:
        def_query_path = get_value(config, 'BENCHMARK', 'queries', '')
        query_path = st.text_input('Path to SQL Queries', value=def_query_path)
        objective = search.objectives.Objective.TIME
    elif benchmark_type == 1:
        def_oltp_home = get_value(config, 'BENCHMARK', 'oltp_home', '')
        def_oltp_config = get_value(config, 'BENCHMARK', 'oltp_config', '')
        def_template_db = get_value(config, 'BENCHMARK', 'template_db', '')
        def_target_db = get_value(config, 'BENCHMARK', 'target_db', '')
        def_reset_every = int(get_value(config, 'BENCHMARK', 'reset_every', 10))
        oltp_home = st.text_input(
            'Home Directory of OLTP Benchmark Generator', value=def_oltp_home)
        oltp_config = st.text_input(
            'Path to OLTP Configuration File', value=def_oltp_config)
        oltp_result = pathlib.Path(oltp_home).joinpath('results')
        reset_every = int(st.number_input(
            'Database Reset Frequency', value=def_reset_every))
        template_db = st.text_input(
            'Name of Template Database', value=def_template_db)
        target_db = st.text_input(
            'Name of Target Database', value=def_target_db)
        objective = search.objectives.Objective.THROUGHPUT
    else:
        raise ValueError(f'Error - unknown benchmark type: {benchmark_type}')


if st.button('Start Tuning'):
    
    if dbms_id == 0:
        dbms = PgConfig(
            db_name, db_user, db_pwd, restart_cmd, 
            recover_cmd, timeout_s)
    elif dbms_id == 1:
        dbms = MySQLconfig(
            db_name, db_user, db_pwd, restart_cmd, 
            recover_cmd, timeout_s)
    else:
        raise ValueError(f'Unknown DBMS ID: {dbms}')
    
    if benchmark_type == 0:
        bench = benchmark.evaluate.OLAP(dbms, query_path)
    elif benchmark_type == 1:
        bench = benchmark.evaluate.TpcC(
            oltp_home, oltp_config, oltp_result, 
            dbms, template_db, target_db, reset_every)
    else:
        raise ValueError(f'Unknown benchmark type: {benchmark_type}')
    
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
    
    st.markdown('### Extracted Tuning Hints')
    hint_rows = []
    for param, doc_hints in docs.param_to_hints.items():
        frequency = len(doc_hints)
        for doc_id, hint in doc_hints:
            row = [param, frequency, doc_id, hint.value.group(), 
                   hint.passage, hint.hint_type.name]
            hint_rows += [row]
    hint_df = pd.DataFrame(hint_rows, columns=[
        'Parameter', 'Frequency', 'Document', 
        'Value', 'Text', 'Inferred Type'])
    st.dataframe(hint_df)
    
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

    st.markdown('### Evaluated DBMS Configurations')
    evaluation_df = pd.DataFrame(columns=[
        'Elapsed (ms)', 'Evaluations', 'Configuration', 
        'Performance', 'Best Configuration', 'Best Performance'],
        index=range(0))
    evaluation_table = st.dataframe(evaluation_df)
    
    st.markdown('### DBMS Tuning Decisions')
    decision_df = pd.DataFrame(columns=[
        'Parameter', 'Recommendation',  'Inferred Type', 
        'Base', 'Factor', 'Value', 'Weight', 
        'Accepted', 'Reward'], index=range(0))
    decision_table = st.dataframe(decision_df)
        
    # Warm-up phase (quick), followed by actual tuning
    st.write(f'Running for up to {timeout_s} seconds, {nr_frames} frames')
    start_s = time.time()
    # Could move warmup to pre-training
    for i in range(nr_frames):
        
        model.learn(total_timesteps=1)
        for log_entry in bench.log:
            evaluation_table.add_rows(log_entry)
        bench.log = []
        for log_entry in unsupervised_env.log:
            decision_table.add_rows(log_entry)
        unsupervised_env.log = []
        
        elapsed_s = time.time() - start_s
        if elapsed_s > timeout_s:
            break
        if i % 500 == 0:
            st.write(f'Step {i} - tuned for {elapsed_s} seconds')
            
    # Show final summary
    st.write('Tuning process of DB-BERT is finished.')
    # st.write('Summary of results:')
    bench.print_stats()