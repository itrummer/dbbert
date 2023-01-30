'''
Created on Jan 30, 2023

@author: immanueltrummer
'''
import os
import pathlib
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
import benchmark
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

config_dir = root_dir.joinpath('demo_configs')
docs_dir = root_dir.joinpath('demo_docs')

for config_path, path_to_docs, log_path in [
    (config_dir.joinpath('pg_tpch_base.ini'), 
        docs_dir.joinpath('postgres100'), 'pg_tpch_log'),
    (config_dir.joinpath('ms_tpch_base.ini'), 
        docs_dir.joinpath('mysql100'), 'ms_tpch_log'),
    (config_dir.joinpath('pg_tpcc_base.ini'), 
        docs_dir.joinpath('postgres100'), 'pg_tpcc_log'),
    (config_dir.joinpath('ms_tpcc_base.ini'), 
        docs_dir.joinpath('mysql100'), 'ms_tpcc_log'),
    (config_dir.joinpath('pg_tpch_base.ini'), 
        docs_dir.joinpath('pg_tpch_single'), 'pg_tpch_single_log')
    ]:
    config_path = str(config_path)
    path_to_docs = str(path_to_docs)
    
    config = ConfigParser()
    config.read(config_path)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    max_length = int(get_value(config, 'BENCHMARK', 'max_length', 128))
    min_batch_size = int(get_value(config, 'BENCHMARK', 'min_batch_size', 8))
    filter_params = int(get_value(config, 'BENCHMARK', 'filter_param', 1))
    use_implicit = int(get_value(config, 'BENCHMARK', 'use_implicit', 1))
    hint_order_id = int(get_value(config, 'BENCHMARK', 'hint_order', 2))
    hint_order = [
        environment.multi_doc.HintOrder.DOCUMENT, 
        environment.multi_doc.HintOrder.BY_PARAMETER, 
        environment.multi_doc.HintOrder.BY_STRIDE][hint_order_id]
    
    nr_frames = int(get_value(config, 'LEARNING', 'nr_frames', 1))
    timeout_s = int(get_value(config, 'LEARNING', 'timeout_s', 120))
    p_scaling = float(get_value(
        config, 'LEARNING', 'performance_scaling', 0.1))
    a_scaling = float(get_value(
        config, 'LEARNING', 'assignment_scaling', 0.1))
    nr_evals = int(get_value(config, 'LEARNING', 'nr_evaluations', 1))
    nr_hints = int(get_value(config, 'LEARNING', 'nr_hints', 1))
    
    memory = float(get_value(config, 'BENCHMARK', 'memory', 8000000))
    disk = float(get_value(config, 'BENCHMARK', 'disk', 500000000))
    cores = int(get_value(config, 'BENCHMARK', 'cores', 8))
    
    def_db = get_value(config, 'DATABASE', 'dbms', 'pg')
    db_name = get_value(config, 'DATABASE', 'name', '')
    db_user = get_value(config, 'DATABASE', 'user', 'ubuntu')
    db_pwd = get_value(config, 'DATABASE', 'password', '')
    restart_cmd = get_value(config, 'DATABASE', 'restart_cmd', '')
    recover_cmd = get_value(config, 'DATABASE', 'recovery_cmd', '')
    if def_db == 'ms':
        def_db_idx = 1
    else:
        def_db_idx = 0
    dbms_id = ['Postgres', 'MySQL'][def_db_idx]
    
    benchmark_type = int(get_value(config, 'BENCHMARK', 'type', 0))
    if benchmark_type == 0:
        query_path = get_value(config, 'BENCHMARK', 'queries', '')
        objective = search.objectives.Objective.TIME
    elif benchmark_type == 1:
        oltp_home = get_value(config, 'BENCHMARK', 'oltp_home', '')
        oltp_config = get_value(config, 'BENCHMARK', 'oltp_config', '')
        template_db = get_value(config, 'DATABASE', 'template_db', '')
        target_db = get_value(config, 'DATABASE', 'target_db', '')
        reset_every = int(get_value(config, 'BENCHMARK', 'reset_every', 10))
        oltp_result = pathlib.Path(oltp_home).joinpath('results')
        objective = search.objectives.Objective.THROUGHPUT
    else:
        raise ValueError(f'Error - unknown benchmark type: {benchmark_type}')
    
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
    
    for run_ctr in range(5):
        # Initialize for new run
        dbms.reset_config()
        dbms.reconfigure()
        bench.reset(log_path, run_ctr)
        
        # Initialize input documents
        docs = DocCollection(
            docs_path=path_to_docs, dbms=dbms, size_threshold=max_length,
            use_implicit=use_implicit, filter_params=filter_params)
        
        hint_rows = []
        for param, doc_hints in docs.param_to_hints.items():
            frequency = len(doc_hints)
            for doc_id, hint in doc_hints:
                row = [doc_id, hint.passage, param, frequency, 
                       hint.value.group(), str(hint.hint_type)]
                hint_rows += [row]
        hint_df = pd.DataFrame(
            hint_rows, columns=[
                'Document', 'Text', 
                'Parameter', 'Frequency',
                'Value', 'Inferred Type'])
        
        # Initialize environment
        set_random_seed(0)
        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(0)
        set_global_seeds(0)
        
        unsupervised_env = NlpTuningEnv(
            docs=docs, max_length=max_length, hint_order=hint_order, 
            dbms=dbms, benchmark=bench, 
            hardware={'memory':memory, 'disk':disk, 'cores':cores}, 
            hints_per_episode=nr_hints, nr_evals=nr_evals, 
            scale_perf=p_scaling, scale_asg=a_scaling, objective=objective)
        unsupervised_env.reset()
        # unsupervised_env = GymEnvironment(unsupervised_env, device=device)
        
        # Initialize agents
        model = A2C(
            'MlpPolicy', unsupervised_env, 
            verbose=1, normalize_advantage=True)
        
        evaluation_df = pd.DataFrame(columns=[
            'Elapsed (ms)', 'Evaluations', 'Configuration', 
            'Performance', 'Best Configuration', 'Best Performance'],
            index=range(0))
            
        # Warm-up phase (quick), followed by actual tuning
        start_s = time.time()
        # Could move warmup to pre-training
        for i in range(nr_frames):
            
            model.learn(total_timesteps=1)            
            elapsed_s = time.time() - start_s
            if elapsed_s > timeout_s:
                break