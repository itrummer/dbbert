'''
Created on Aug 15, 2023

@author: immanueltrummer
'''
from pybullet_utils.util import set_global_seeds

import argparse
import benchmark.factory
import dbms.factory
import numpy as np
import random
import time
import torch


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'text_source_path', type=str, help='Path to input text')
    parser.add_argument(
        '--max_length', type=int, default=128, 
        help='Maximal length of text chunk in characters')
    parser.add_argument(
        '--filter_params', type=int, default=1, choices={0, 1},
        help='Set to 1 to filter text passages using heuristics')
    parser.add_argument(
        '--use_implicit', type=int, default=1, choices={0, 1},
        help='Set to 1 to recognize implicit parameter references')
    parser.add_argument(
        '--hint_order', type=int, default=2, choices={0, 1, 2},
        help='Order hints by document (0), parameter (1), or optimized (2)')
    parser.add_argument(
        '--nr_frames', type=int, default=100000, help='Maximal number of frames')
    parser.add_argument(
        '--timeout_s', type=int, default=1500, help='Tuning timeout in seconds')
    parser.add_argument(
        '--performance_scaling', type=float, default=0.1,
        help='Scaling factor for performance-related rewards')
    parser.add_argument(
        '--assignment_scaling', type=float, default=1.0,
        help='Scaling factor for rewards due to successful value assignments')
    parser.add_argument(
        '--nr_evaluations', type=int, default=2,
        help='Number of parameter settings to try per batch of tuning hints')
    parser.add_argument(
        '--nr_hints', type=int, default=20,
        help='Number of hints to consider when selecting parameter settings')
    parser.add_argument(
        '--min_batch_size', type=int, default=8,
        help='Batch size when processing text via language models')
    parser.add_argument(
        'memory', type=int, default=8000000,
        help='Main memory of target system, measured in bytes')
    parser.add_argument(
        'disk', type=int, default=100000000,
        help='Disk space of target system, measured in bytes')
    parser.add_argument(
        'cores', type=int, default=8, help='Number of cores of target system')
    parser.add_argument(
        'dbms', type=str, choices={'pg', 'ms'},
        help='Set to "pg" to tune PostgreSQL, "ms" to tune MySQL')
    parser.add_argument('db_name', type=str, help='Name of database to tune')
    parser.add_argument('db_user', type=str, help='Name of database login')
    parser.add_argument('db_pwd', type=str, help='Password for database login')
    parser.add_argument(
        'restart_cmd', type=str, 
        help='Terminal command for restarting database server')
    parser.add_argument(
        '--recover_cmd', type=str, 
        default='echo "Reset database state!"; sleep 5',
        help='Command to restore default status of database system')
    parser.add_argument(
        'query_path', type=str, default=None, 
        help='Path to file containing SQL queries')
    parser.add_argument(
        '--nr_runs', type=int, default=1, help='Number of benchmark runs')
    parser.add_argument(
        '--result_path_prefix', type=str, default='dbbert_results',
        help='Path prefix for files containing tuning results')
    args = parser.parse_args()
    print(f'Input arguments: {args}')

    # Expensive import statements after parsing arguments
    from environment.zero_shot import NlpTuningEnv
    from stable_baselines3 import A2C
    from doc.collection import DocCollection
    from stable_baselines3.common.utils import set_random_seed
    import environment.multi_doc
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hint_order = [
        environment.multi_doc.HintOrder.DOCUMENT, 
        environment.multi_doc.HintOrder.BY_PARAMETER, 
        environment.multi_doc.HintOrder.BY_STRIDE][args.hint_order]
    
    dbms = dbms.factory.from_args(args)
    objective, bench = benchmark.factory.from_args(args, dbms)
    
    for run_ctr in range(args.nr_runs):
        # Initialize for new run
        dbms.reset_config()
        dbms.reconfigure()
        bench.reset(args.result_path_prefix, run_ctr)
        
        # Initialize input documents
        docs = DocCollection(
            docs_path=args.text_source_path, dbms=dbms, 
            size_threshold=args.max_length,
            use_implicit=args.use_implicit, 
            filter_params=args.filter_params)
        
        # Initialize environment
        set_random_seed(0)
        random.seed(1)
        np.random.seed(1)
        torch.manual_seed(0)
        set_global_seeds(0)
        hardware = {'memory':args.memory, 'disk':args.disk, 'cores':args.cores}
        unsupervised_env = NlpTuningEnv(
            docs=docs, max_length=args.max_length, hint_order=hint_order, 
            dbms=dbms, benchmark=bench, hardware=hardware, 
            hints_per_episode=args.nr_hints, nr_evals=args.nr_evaluations, 
            scale_perf=args.performance_scaling, 
            scale_asg=args.assignment_scaling, objective=objective)
        unsupervised_env.reset()
        
        # Initialize agents
        model = A2C(
            'MlpPolicy', unsupervised_env, 
            verbose=1, normalize_advantage=True)
        
        # Start benchmark run
        start_s = time.time()
        for i in range(args.nr_frames):
            model.learn(total_timesteps=1)            
            elapsed_s = time.time() - start_s
            if elapsed_s > args.timeout_s:
                break