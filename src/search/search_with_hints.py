'''
Created on Apr 16, 2021

@author: immanueltrummer
'''
from collections import defaultdict
from dbms.generic_dbms import ConfigurableDBMS
from benchmark.evaluate import Benchmark

class ParameterExplorer():
    """ Explores the parameter space using previously collected tuning hints. """

    def __init__(self, dbms: ConfigurableDBMS, benchmark: Benchmark):
        """ Initializes for given benchmark and database system. 
        
        Args:
            dbms: explore parameters of this database system.
            benchmark: optimize parameters for this benchmark.
        """
        self.dbms = dbms
        self.benchmark = benchmark
        self.def_millis = self._def_conf_millis()
        
    def _def_conf_millis(self):
        """ Returns milliseconds for running benchmark with default configuration. """
        self.dbms.reset_config()
        self.dbms.reconfigure()
        _, def_millis = self.benchmark.evaluate(self.dbms)
        return def_millis
        
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
        max_savings = 0
        best_config = {}
        for config in configs:
            savings = self._evaluate_config(config)
            if savings > max_savings:
                max_savings = savings
                best_config = config
        print(f'Obtained {max_savings} by configuration {best_config}')
        return max_savings, best_config

    def _select_configs(self, hint_to_weight, nr_evals):
        """ Returns set of interesting configurations, based on hints. 
        
        Args:
            hint_to_weight: maps tuning hints to a weight
            nr_evals: select that many configurations
            
        Returns:
            
        """
        # Sort hints by their weight (decreasing)
        sorted_hints = dict(sorted(hint_to_weight.items(), key=lambda item: -item[1]))
        nr_hints = len(sorted_hints)
        if nr_hints <= nr_evals:
            return [{k[0]: k[1]} for k, _ in sorted_hints.items()]
        else:
            return [{k[0]: k[1]} for k, _ in list(sorted_hints.items())[:nr_evals]]
        
    def _gather_values(self, hint_to_weight):
        """ Gather alternative values for the same parameter. 
        
        Returns:
            Dictionary mapping parameters to lists of values
        """
        param_to_vals = defaultdict(lambda: [])
        for hint in hint_to_weight:
            param = hint.param.group()
            value = hint.value.group()
            param_to_vals[param] += [value]
        return param_to_vals
    
    def _evaluate_config(self, config):
        """ Evaluates given configuration and returns duration in milliseconds. 
        
        Args:
            config: dictionary mapping parameters to values
        
        Returns:
            Improvement over default configuration in milliseconds.
        """
        self.dbms.reset_config()
        print(f'Trying configuration: {config}')
        for param, value in config.items():
            self.dbms.set_param_smart(param, value, 1)
        self.dbms.reconfigure()
        error, millis = self.benchmark.evaluate(self.dbms)
        savings = millis - self.def_millis
        print(f'Saved {savings} millis with {config}')
        return savings if not error else -10000