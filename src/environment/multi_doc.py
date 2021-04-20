'''
Created on Apr 18, 2021

@author: immanueltrummer
'''
from environment.common import DocTuning
from benchmark.evaluate import Benchmark
from environment.common import DecisionType
from collections import defaultdict
from search.search_with_hints import ParameterExplorer
from doc.collection import DocCollection
from dbms.generic_dbms import ConfigurableDBMS

class MultiDocTuning(DocTuning):
    """ Agent finds good configurations by aggregating tuning document collections. """

    def __init__(self, docs: DocCollection, dbms: ConfigurableDBMS, benchmark: Benchmark,
                 hardware, nr_hints, nr_rereads, nr_evals):
        """ Initialize from given tuning documents, database, and benchmark. 
        
        Args:
            docs: collection of text documents with tuning hints
            dbms: database management system to tune
            benchmark: benchmark for which to tune system
            hardware: memory size, disk size, and number of cores
            nr_hints: how many hints to consider
            nr_rereads: how often to read the hints
            nr_evals: how many evaluations with extracted hints
        """
        super().__init__(docs)
        self.dbms = dbms
        self.benchmark = benchmark
        self.hardware = hardware
        self.nr_rereads = nr_rereads
        self.nr_evals = nr_evals
        self.hints = self._ordered_hints()
        self.nr_hints = min(nr_hints, len(self.hints))
        print('All hints considered for multi-doc tuning:')
        for i in range(self.nr_hints):
            _, hint = self.hints[i]
            print(f'Hint nr. {i}: {hint.param} -> {hint.value}')
        self.explorer = ParameterExplorer(dbms, benchmark)
        self.reset()
        
    def _ordered_hints(self):
        """ Prioritize hints in tuning document collection. """
        ordered_hints = []
        for param, _ in self.docs.param_counts.most_common():
            param_hints = self.docs.param_to_hints[param]
            ordered_hints += param_hints
        return ordered_hints

    def _take_action(self, action):
        """ Process action and return obtained reward. """
        reward = 0
        _, hint = self.hints[self.hint_ctr]
        # Distinguish by decision type
        if self.decision == DecisionType.PICK_BASE:
            if action <= 2 and hint.float_val < 1.0:
                # Multiply given value with hardware properties
                self.base = float(self.hardware[action]) * hint.float_val
            else:
                # Use provided value as is
                self.base = hint.float_val
        elif self.decision == DecisionType.PICK_FACTOR:
            self.factor = float(self.factors[action])
        else:
            reward = self._process_hint(hint, action)
        return reward
    
    def _process_hint(self, hint, action):
        """ Finishes processing current hint and returns direct reward. """
        param = hint.param.group()
        value = str(int(self.base * self.factor)) + hint.val_unit 
        print(f'Trying to set {param} to {value} ...')
        success = self.dbms.can_set(param, value)
        assignment = (param, value)
        if success:
            reward = 10
            weight = pow(2, action)
            self.hint_to_weight[assignment] += weight
            print(f'Success! Choosing weight {weight} for {assignment}.')
        else:
            reward = -10
            print(f'Failed assignment: {assignment}.')
        return reward        

    def _finalize_episode(self):
        """ Return optimal benchmark time when using weighted hints. """
        savings, config = self.explorer.explore(self.hint_to_weight, self.nr_evals)
        print(f'Achieved maximal savings of {savings} millis using {config}')
        return savings

    def _reset(self):
        """ Initializes for new tuning episode. """
        self.label = None
        self.hint_to_weight = defaultdict(lambda: 0)
        self.benchmark.print_stats()