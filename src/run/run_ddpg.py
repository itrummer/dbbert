'''
Created on Aug 17, 2023

@author: immanueltrummer
'''
'''
Created on Apr 26, 2021

@author: immanueltrummer
'''
import analysis.simulation
import argparse
import benchmark.factory
import dbms.factory
import numpy as np
import random
import torch

from analysis.util import get_analysis_logger, TimerStruct  # noqa
from analysis.ddpg.ddpg import DDPG  # noqa
from parameters.util import decompose_val, is_numerical
from search.objectives import calculate_reward


class DDPGenv(object):
    """ Environment teaching DDPG++ agent how to tune a DBMS. """
    
    def __init__(self, dbms, benchmark, objective, knob_names, max_val_change):
        """ Initializes tuning environment for specific tuning knobs. 
        
        Args:
            dbms: link to database management system.
            benchmark: optimize configuration for this benchmark.
            objective: performance goal of database tuning.
            knob_names: names of tuning knobs to consider.
            max_val_change: change default values by at most that factor.
        """
        self.dbms = dbms
        self.benchmark = benchmark
        self.objective = objective
        self.knob_names = knob_names
        self.max_val_change = float(max_val_change)
        self.knob_dim = len(self.knob_names)
        self.knobs_min = []
        self.knobs_max = []
        self.knob_units = []
        self.metric_dim = 1
        self._set_val_ranges()
        self.dbms.reset_config()
        self.dbms.reconfigure()
        self.def_metrics = self.benchmark.evaluate()
        
    def _set_val_ranges(self):
        """ Sets minimal and maximal knob values to consider for tuning. """
        self.dbms.reset_config()
        self.dbms.reconfigure()
        for knob in self.knob_names:
            raw_val = self.dbms.get_value(knob)
            print(f'knob: {knob}; raw_val: {raw_val}')
            if raw_val in ['on', 'off', '1', '0', 1, 0]:
                unit = ''
                min_val = 0
                max_val = 1
            else:
                f_val, unit = decompose_val(raw_val)
                min_val = int(f_val / self.max_val_change)
                max_val = int(f_val* self.max_val_change)
                
            self.knobs_min.append(min_val)
            self.knobs_max.append(max_val)
            self.knob_units.append(unit)

    def evaluate(self, knob_data):
        """ Evaluate current configuration.
        
        Args:
            knob_data: settings for all tuning knobs.
        
        Returns:
            reward values (twice).
        """
        self.dbms.reset_config()
        for i in range(self.knob_dim):
            knob = self.knob_names[i]
            min_ = self.knobs_min[i]
            max_ = self.knobs_max[i]
            int_val = int(round(min_ + (max_ - min_) * knob_data[i]))
            unit = self.knob_units[i]
            value = str(int_val) + unit
            print(f'Setting {knob} to {value}')
            success = self.dbms.set_param_smart(knob, value)
            print(f'Set successful: {success}')
        self.dbms.reconfigure()
        metrics = self.benchmark.evaluate()
        reward_val = calculate_reward(metrics, self.def_metrics, self.objective)
        reward_array = np.array([reward_val])
        return reward_array, reward_array


def run_ddpg(dbms, benchmark, objective, knob_names, max_val_change, timeout_s):
    """ Run benchmark for DDPG+ algorithm, using specified knobs and ranges. 
    
    Args:
        dbms: represents database management system to tune.
        benchmark: benchmark to optimize performance for.
        objective: performance optimization objective (e.g., minimize time).
        knob_names: names of tuning knobs.
        max_val_change: maximal relative deviation from default values.
        timeout_s: tuning timeout in seconds.
    """
    random.seed(1)
    np.random.seed(1)
    torch.manual_seed(0)
    env = DDPGenv(dbms, benchmark, objective, knob_names, max_val_change)
    ddpg_config = {'gamma': 0., 'c_lr': 0.001, 'a_lr': 0.02, 
                   'num_collections': 2, 'n_epochs': 30, 
                   'a_hidden_sizes': [128, 128, 64], 
                   'c_hidden_sizes': [64, 128, 64]}
    analysis.simulation.ddpg(
        env, ddpg_config, n_loops=200, timeout_s=timeout_s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--tolerance', type=float, 
        help='Deviate from default values by at most this factor')
    parser.add_argument(
        '--timeout_s', type=int, default=1500, help='Tuning timeout in seconds')
    parser.add_argument(
        '--nr_evaluations', type=int, default=2,
        help='Number of parameter settings to try per batch of tuning hints')
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
    
    dbms = dbms.factory.from_args(args)
    objective, bench = benchmark.factory.from_args(args, dbms)

    for run_ctr in range(args.nr_runs):
        
        print(f'Preparing for run number {run_ctr} ...')
        dbms.reset_config()
        dbms.reconfigure()
        all_params = dbms.all_params()
        all_params = [p for p in all_params if is_numerical(dbms.get_value(p))]
        
        print(f'Starting run number {run_ctr} ...')
        bench.reset(args.result_path_prefix, run_ctr)
        run_ddpg(
            dbms, bench, objective, all_params, 
            args.tolerance, args.timeout_s)