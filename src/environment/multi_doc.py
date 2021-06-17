'''
Created on Apr 18, 2021

@author: immanueltrummer
'''
from environment.bert_tuning import TuningBertFine
from benchmark.evaluate import Benchmark
from environment.common import DecisionType
from collections import defaultdict
from search.search_with_hints import ParameterExplorer
from doc.collection import DocCollection
from dbms.generic_dbms import ConfigurableDBMS
import enum

class HintOrder(enum.IntEnum):
    """ The order in which tuning hints are considered. """
    DOCUMENT=0, # process hints in document order
    BY_PARAMETER=1, # prioritize hints about frequently mentioned parameters
    BY_STRIDE=2, # round robin over hint batches associated with parameters
    
def parse_order(config):
    """ Parse hint order from configuration file. """
    order_str = config['BENCHMARK']['hint_order']
    if order_str == 'by_parameter':
        print('Sorting hints by parameter')
        return HintOrder.BY_PARAMETER
    elif order_str == 'by_stride':
        print('Sorting hints by stride')
        return HintOrder.BY_STRIDE
    else:
        print('Hints in document order')
        return HintOrder.DOCUMENT

class MultiDocTuning(TuningBertFine):
    """ Agent finds good configurations by aggregating tuning document collections. """

    def __init__(
            self, docs: DocCollection, max_length, mask_params, hint_order,
            dbms: ConfigurableDBMS, benchmark: Benchmark, hardware, 
            hints_per_episode, nr_evals, scale_perf, scale_asg, objective):
        """ Initialize from given tuning documents, database, and benchmark. 
        
        Args:
            docs: collection of text documents with tuning hints
            max_length: maximum number of tokens per snippet
            mask_params: whether to mask parameters or not
            hint_order: process tuning hints in this order
            dbms: database management system to tune
            benchmark: benchmark for which to tune system
            hardware: memory size, disk size, and number of cores
            hints_per_episode: candidate hints before episode ends
            nr_evals: how many evaluations with extracted hints
            scale_perf: scale performance reward by this factor
            scale_asg: scale reward for successful assignments
            objective: describes the optimization goal
        """
        super().__init__(docs, hints_per_episode, max_length, mask_params)
        self.dbms = dbms
        self.benchmark = benchmark
        self.hardware = hardware
        self.nr_evals = nr_evals
        self.scale_perf = scale_perf
        self.scale_asg = scale_asg
        self.docs.doc_to_hints
        self.hints = self._ordered_hints(hint_order)
        self.nr_hints = len(self.hints)
        if hints_per_episode == -1:
            self.hints_per_episode = self.nr_hints
        else:
            self.hints_per_episode = hints_per_episode
        print('All hints considered for multi-doc tuning:')
        for i in range(self.nr_hints):
            _, hint = self.hints[i]
            print(f'Hint nr. {i}: {hint.param.group()} -> {hint.value.group()}')
        self.explorer = ParameterExplorer(dbms, benchmark, objective)
        self.reset()
        
    def _ordered_hints(self, hint_order):
        """ Returns hints according to specified order. """
        if hint_order == HintOrder.BY_PARAMETER:
            return self._hints_by_param()
        elif hint_order == HintOrder.BY_STRIDE:
            return self._hints_by_stride()
        else:
            return self._hints_by_doc()

    def _hints_by_doc(self):
        """ Returns hints in document collection order. """
        hints = []
        for doc_id in range(self.docs.nr_docs):
            hints += [(doc_id, hint) for hint in self.docs.get_hints(doc_id)]
        return hints
        
    def _hints_by_param(self):
        """ Order hints by occurrence frequency of associated parameter. """
        ordered_hints = []
        for param, _ in self.docs.param_counts.most_common():
            param_hints = self.docs.param_to_hints[param]
            ordered_hints += param_hints
        return ordered_hints

    def _hints_by_stride(self):
        """ Round robin between parameters based on occurrence frequency. """
        ordered_hints = []
        hints_per_param = max([len(h) for h in self.docs.param_to_hints.values()])
        param_to_list = {p:list(v) for p, v in self.docs.param_to_hints.items()}
        step = 10
        for lb in range(0, hints_per_param, step):
            for param, _ in self.docs.param_counts.most_common():
                param_hints = param_to_list[param]
                nr_param_hints = len(param_hints)
                if lb < nr_param_hints:
                    ub = min(lb + step, nr_param_hints-1)
                    stride = param_hints[lb:ub]
                    ordered_hints += stride
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
        success = self.dbms.can_set(param, value)
        assignment = (param, value)
        print(f'Trying assigning {param} to {value}')
        if success:
            reward = 10 * self.scale_asg
            weight = pow(2, action)
            self.hint_to_weight[assignment] += weight
            print(f'Adding assignment {assignment} with weight {weight}')
            print(f'Assignment {assignment} extracted from "{hint.passage}"')
        else:
            reward = -10
        return reward

    def _finalize_episode(self):
        """ Return optimal benchmark reward when using weighted hints. """
        if self.hint_to_weight:
            reward, config = self.explorer.explore(
                self.hint_to_weight, self.nr_evals)
            print(f'Achieved unscaled reward of {reward} using {config}')
            return reward * self.scale_perf
        else:
            return 0

    def _reset(self):
        """ Initializes for new tuning episode. """
        self.label = None
        self.hint_to_weight = defaultdict(lambda: 0)
        self.benchmark.print_stats()