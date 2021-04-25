'''
Created on Apr 16, 2021

@author: immanueltrummer
'''
from collections import defaultdict
from dbms.generic_dbms import ConfigurableDBMS
from benchmark.evaluate import Benchmark
from parameters.util import is_numerical, decompose_val
from search.objectives import calculate_reward

class ParameterExplorer():
    """ Explores the parameter space using previously collected tuning hints. """

    def __init__(self, dbms: ConfigurableDBMS, benchmark: Benchmark, objective):
        """ Initializes for given benchmark and database system. 
        
        Args:
            dbms: explore parameters of this database system.
            benchmark: optimize parameters for this benchmark.
            objective: goal of parameter optimization.
        """
        self.dbms = dbms
        self.benchmark = benchmark
        self.def_metrics = self._def_conf_metrics()
        self.objective = objective
        
    def _def_conf_metrics(self):
        """ Returns metrics for running benchmark with default configuration. """
        if self.dbms and self.benchmark:
            self.dbms.reset_config()
            self.dbms.reconfigure()
            def_metrics = self.benchmark.evaluate()
        else:
            print('Warning: no DBMS or benchmark specified for parameter exploration.')
            def_metrics = {'error': False, 'time': 0}
        return def_metrics
        
    def explore(self, hint_to_weight, nr_evals):
        """ Explore parameters to improve benchmark performance.
        
        Args:
            hint_to_weight: use weighted hints as guidelines for exploration
            nr_evals: evaluate so many parameter configurations
        
        Returns:
            Returns maximal improvement and associated configuration
        """
        print(f'Weighted hints: {hint_to_weight}')
        configs = self._select_configs(hint_to_weight, nr_evals)
        print(f'Selected configurations: {configs}')
        # Identify best configuration
        max_reward = 0
        best_config = {}
        for config in configs:
            reward = self._evaluate_config(config)
            if reward > max_reward:
                max_reward = reward
                best_config = config
        print(f'Obtained {max_reward} by configuration {best_config}')
        return max_reward, best_config

    def _select_configs(self, hint_to_weight, nr_evals):
        """ Returns set of interesting configurations, based on hints. 
        
        Args:
            hint_to_weight: maps assignments to a weight
            nr_evals: select that many configurations
            
        Returns:
            List of configurations to try out
        """
        param_to_w_vals = self._gather_values(hint_to_weight)
        configs = []
        for _ in range(nr_evals):
            config = self._next_config(configs, param_to_w_vals)
            configs.append(config)
        return configs
         
    def _next_config(self, configs, param_to_w_vals):
        """ Select most interesting configuration to try next. """
        config = {}
        for p, w_vals in param_to_w_vals.items():
            ref_vals = [c[p] for c in configs]
            vals = [v for v, _ in w_vals]
            min_dist = float('inf')
            best_val = vals[0]
            for val in vals:
                exp_refs = ref_vals + [val]
                exp_dist = self._max_min_distance(w_vals, exp_refs)
                if exp_dist < min_dist:
                    min_dist = exp_dist
                    best_val = val
            config[p] = best_val
        return config
         
    def _max_min_distance(self, weighted_values, ref_vals):
        """ Returns maximum of minimal distances to reference values. """
        return max([w * self._min_distance(v, ref_vals) for v, w in weighted_values])
        
    def _min_distance(self, value, ref_vals):
        """ Returns minimum distance to value over all reference values. """
        return min([self._distance(value, r) for r in ref_vals])
        
    def _distance(self, value_1, value_2):
        """ Calculate raw distance between two assignments for same value. 
        
        Args:
            value_1: first assignment value
            value_2: second assignment value
            
        Returns:
            distance between assignment values
        """
        if value_1 == value_2:
            return 0
        elif is_numerical(value_1) and is_numerical(value_2):
            float_1, unit_1 = decompose_val(value_1)
            float_2, unit_2 = decompose_val(value_2)
            if unit_1 == unit_2:
                return abs(float_1 - float_2)
            else:
                return 1000
        else:
            return 10000       
        
    def _gather_values(self, hint_to_weight):
        """ Gather weighted value suggestions for the same parameter.
        
        Args:
            hint_to_weight: maps parameter-value assignments to weights
        
        Returns:
            Dictionary mapping parameters to lists of value-weight pairs
        """
        param_to_vals = defaultdict(lambda: [])
        for (param, value), weight in hint_to_weight.items():
            param_to_vals[param] += [(value, weight)]
        return param_to_vals
    
    def _evaluate_config(self, config):
        """ Evaluates given configuration and returns duration in milliseconds. 
        
        Args:
            config: dictionary mapping parameters to values
        
        Returns:
            Improvement over default configuration in milliseconds.
        """
        if self.dbms:
            self.dbms.reset_config()
            print(f'Trying configuration: {config}')
            for param, value in config.items():
                self.dbms.set_param_smart(param, value)
            self.dbms.reconfigure()
            metrics = self.benchmark.evaluate()
            reward = calculate_reward(metrics, self.def_metrics, self.objective)
            print(f'Reward {reward} with {config}')
            return reward
        else:
            return 0