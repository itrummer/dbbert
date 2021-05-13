'''
Created on Apr 26, 2021

@author: immanueltrummer
'''
import argparse
import numpy as np
#import sys
#sys.path.append('/Users/immanueltrummer/git/ottertune/server')
from analysis.util import get_analysis_logger, TimerStruct  # noqa
from analysis.ddpg.ddpg import DDPG  # noqa
import analysis.simulation
import configparser.ConfigParser
import benchmark.factory
import dbms.factory
from parameters.util import read_numerical, is_numerical
import random
import search.objectives
import torch

from parameters.util import decompose_val
from search.objectives import calculate_reward
from dbms.postgres import PgConfig

class DDPGenv(object):
    """ Environment teaching DDPG++ agent how to tune a DBMS. """
    
    def __init__(self, dbms, benchmark, objective, knob_names, max_val_change):
        """ Initializes tuning environment for specific tuning knobs. 
        
        Args:
            dbms: link to database management system
            benchmark: optimize configuration for this benchmark
            objective: performance goal of database tuning
            knob_names: names of tuning knobs to consider
            max_val_change: change default values by at most that factor
            dbms: configurable database management system
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
        """ Sets minimal and maximal knob values to consider during exploration. """
        self.dbms.reset_config()
        self.dbms.reconfigure()
        for knob in self.knob_names:
            raw_val = self.dbms.get_value(knob)
            print(f'knob: {knob}; raw_val: {raw_val}')
            f_val, unit = decompose_val(raw_val)
            min_val = int(f_val / self.max_val_change)
            max_val = int(f_val* self.max_val_change)
            self.knobs_min.append(min_val)
            self.knobs_max.append(max_val)
            self.knob_units.append(unit)

    def evaluate(self, knob_data):
        """ Evaluate current configuration. """
        self.dbms.reset_config()
        for i in range(self.knob_dim):
            knob = self.knob_names[i]
            min_ = self.knobs_min[i]
            max_ = self.knobs_max[i]
            int_val = int(min_ + (max_ - min_) * knob_data[i])
            unit = self.knob_units[i]
            value = str(int_val) + unit
            #print(f'Setting {knob} to {value}')
            success = self.dbms.set_param_smart(knob, value)
            #print(f'Set successful: {success}')
        self.dbms.reconfigure()
        metrics = self.benchmark.evaluate()
        reward_val = calculate_reward(metrics, self.def_metrics, self.objective)
        reward_array = np.array([reward_val])
        return reward_array, reward_array

def run_ddpg(dbms, benchmark, objective, knob_names, max_val_change, timeout_s):
    """ Run benchmark for DDPG+ algorithm, using specified knobs and ranges. """
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
    print('Preparing to run DDPG baseline ...')
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to configuration file')
    parser.add_argument('tolerance', type=float, help='Tolerance around default values')
    args = parser.parse_args()
    
    config = configparser.ConfigParser()
    config.read_file(args.config)
    
    dbms = dbms.factory.from_file(config)
    bench = benchmark.factory.from_file(config, dbms)
    objective = search.objectives.from_file(config)
    
    if isinstance(dbms, PgConfig):
        conf_path = config['DATABASE']['config']
        all_params = read_numerical(conf_path)
    else:
        ms_p_vals = dbms.all_params()
        all_params = [p for p, v in ms_p_vals if is_numerical(v)]

    tolerance = args.tolerance
    run_ddpg(dbms, benchmark, objective, all_params, tolerance, 300)