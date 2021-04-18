'''
Created on Apr 2, 2021

@author: immanueltrummer
'''
import gym
import nlp.nlp_util as nlp
import numpy as np
from benchmark.evaluate import Benchmark
from environment.common import DecisionType
from collections import defaultdict
from gym.spaces import Box
from gym.spaces import Discrete
from parameters.util import scale
from search.search_with_hints import ParameterExplorer
import re
import torch
from doc.collection import DocCollection, TuningHint
from dbms.generic_dbms import ConfigurableDBMS

class MultiDocTuning(gym.Env):
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
        self.docs = docs
        self.dbms = dbms
        self.benchmark = benchmark
        self.hardware = hardware
        self.nr_hints = nr_hints
        self.nr_rereads = nr_rereads
        self.nr_evals = nr_evals
        self.ordered_hints = self._ordered_hints()
        self.factors = [0.25, 0.5, 1, 2, 4]
        self.explorer = ParameterExplorer(dbms, benchmark)
        self.observation_space = Box(low=-10, high=10, 
                                     shape=(1537,), dtype=np.float32)
        self.action_space = Discrete(5)
        self.def_obs = torch.zeros(1537)
        self.obs_cache = {}
        self.reset()
        
    def step(self, action):
        """ Potentially apply hint and proceed to next one. """
        reward = self._take_action(action)
        done = self._next_state(action)
        if done:
            reward += self._benchmark()
        return self._observe(), reward, done, {}

    def _take_action(self, action):
        """ Process action and return obtained reward. """
        reward = 0
        _, hint = self.ordered_hints[self.hint_ctr]
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
            param = hint.param.group()
            value = '\'' + str(int(self.base * self.factor)) + hint.val_unit + '\''
            print(f'Trying to set {param} to {value} ...')
            success = self.dbms.can_set(param, value)
            if success:
                reward = 10
                weight = pow(2, action)
                assignment = (param, value)
                self.hint_to_weight[assignment] += weight
                print(f'Success! Choosing weight {weight} for {assignment}.')
            else:
                reward = -10
        return reward
    #
    # def _get_number(self, value):
        # """ Extract number from parameter value string. """
        # if re.match('\d+$', value):
            # return int(value)
        # elif re.match('\d+%$', value):
            # percentage = int(re.sub('(\d+)(%)', '\g<1>', value))
            # return percentage/100.0
        # elif re.match('\d+\w+$', value):
            # number = int(re.sub('(\d+)(\w+)', '\g<1>', value))
            # return number
        # else:
            # return int(1)

    def _next_state(self, action):
        """ Advance to next state in MDP and return termination flag. """
        # print(f'Decision: {self.decision};')
        # print(f'Hint: {self.hint_ctr};')
        # print(f'Re-reads: {self.reread_ctr}')
        done = False
        # Update decision and hint counter
        if self.decision == DecisionType.PICK_BASE:
            if action == 4: # Skip to next hint
                self.hint_ctr += 1
            else:
                self.decision = DecisionType.PICK_FACTOR
        elif self.decision == DecisionType.PICK_FACTOR:
            self.decision = DecisionType.PICK_WEIGHT
        else:
            self.decision = DecisionType.PICK_BASE
            self.hint_ctr += 1
        # Update rereads and termination flag
        if self.hint_ctr >= self.nr_hints:
            self.reread_ctr += 1
        if self.reread_ctr >= self.nr_rereads:
            done = True
        return done

    def _benchmark(self):
        """ Return optimal benchmark time when using weighted hints. """
        savings, config = self.explorer.explore(self.hint_to_weight, self.nr_evals)
        print(f'Achieved maximal savings of {savings} millis using {config}')
        return savings

    def _observe(self):
        """ Generates observations based on current hint. """
        obs = self.def_obs
        if self.hint_ctr < self.nr_hints:
            if self.hint_ctr in self.obs_cache:
                return self.obs_cache[self.hint_ctr]
            else:
                _, hint = self.ordered_hints[self.hint_ctr]
                hint_obs = self._hint_to_obs(hint)
                decision_obs = torch.tensor([int(self.decision)])
                obs = torch.cat((hint_obs, decision_obs))
                self.obs_cache[self.hint_ctr] = obs
        return obs

    def _hint_to_obs(self, hint: TuningHint):
        """ Maps tuning hint to an observation vector. """
        tokens = nlp.tokenize(hint.passage)
        encoding = nlp.encode(hint.passage)
        # Map parameter and value to vector
        obs_parts = []
        for item in [hint.param, hint.value]:
            obs_parts.append(
                nlp.mean_encoding(
                    tokens, encoding, item.start(), item.end()))
        # Use zeros in case of missing vectors
        obs = self.def_obs
        if not (obs_parts[0] is None or obs_parts[1] is None):
            obs = torch.cat((obs_parts[0], obs_parts[1]))
        return obs

    def _ordered_hints(self):
        """ Prioritize hints in tuning document collection. """
        ordered_hints = []
        for param, _ in self.docs.param_counts.most_common():
            param_hints = self.docs.param_to_hints[param]
            ordered_hints += param_hints
        return ordered_hints

    def reset(self):
        """ Initializes for new tuning episode. """
        self.reread_ctr = 0
        self.hint_ctr = 0
        self.decision = DecisionType.PICK_BASE
        self.base = None
        self.factor = None
        self.hint_to_weight = defaultdict(lambda: 0)
        self.benchmark.print_stats()
        return self._observe()