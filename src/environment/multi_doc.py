'''
Created on Apr 2, 2021

@author: immanueltrummer
'''
import gym
import nlp.nlp_util as nlp
import numpy as np
from search.search_with_hints import search_improvements
from benchmark.evaluate import Benchmark
from collections import Counter
from collections import defaultdict
from gym.spaces import Box
from gym.spaces import Discrete
import torch
from random import randint, choices
from doc.collection import DocCollection, TuningHint
from dbms.generic_dbms import ConfigurableDBMS

class MultiDocTuning(gym.Env):
    """ Agent finds good configurations by aggregating tuning document collections. """

    def __init__(self, docs: DocCollection, dbms: ConfigurableDBMS, benchmark: Benchmark,
                 nr_hints, nr_rereads, nr_evals):
        """ Initialize from given tuning documents, database, and benchmark. 
        
        Args:
            docs: collection of text documents with tuning hints
            dbms: database management system to tune
            benchmark: benchmark for which to tune system
            nr_hints: how many hints to consider
            nr_rereads: how often to read the hints
            nr_evals: how many evaluations with extracted hints
        """
        self.docs = docs
        self.dbms = dbms
        self.benchmark = benchmark
        self.nr_hints = nr_hints
        self.nr_rereads = nr_rereads
        self.nr_evals = nr_evals
        self.ordered_hints = self._ordered_hints()
        self.hint_to_weight = defaultdict(lambda: 0)
        self.observation_space = Box(low=-10, high=10, 
                                     shape=(1536,), dtype=np.float32)
        self.action_space = Discrete(5)
        self.def_obs = torch.zeros(1536)
        self.obs_cache = {}
        self.reset()
        
    def step(self, action):
        """ Potentially apply hint and proceed to next one. """
        reward = self._take_action(action)
        done = self._next_state()
        if done:
            reward += self._benchmark()
        return self._observe(), reward, done, {}

    def _take_action(self, action):
        """ Process action and return obtained reward. """
        reward = 0
        # Do we take hint?
        if action>0:
            _, hint = self.ordered_hints[self.hint_ctr]
            param = hint.param.group()
            value = hint.value.group()
            success = self.dbms.can_set(param, value, 1)
            if success:
                reward = 10
                weight = pow(2, action)
                assignment = (param, value)
                self.hint_to_weight[assignment] += weight
            else:
                reward = -10
        return reward

    def _next_state(self):
        """ Advance to next state in MDP and return termination flag. """
        print(f'Hint: {self.hint_ctr}; Re-reads: {self.reread_ctr}')
        done = False
        self.hint_ctr += 1
        if self.hint_ctr >= self.nr_hints:
            self.reread_ctr += 1
        if self.reread_ctr >= self.nr_rereads:
            done = True
        return done

    def _benchmark(self):
        """ Return optimal benchmark time when using weighted hints. """
        return search_improvements(self.dbms, self.benchmark, 
                                   self.hint_to_weight, self.nr_evals)

    def _observe(self):
        """ Generates observations based on current hint. """
        if self.hint_ctr < self.nr_hints:
            if self.hint_ctr in self.obs_cache:
                return self.obs_cache[self.hint_ctr]
            else:
                _, hint = self.ordered_hints[self.hint_ctr]
                obs = self._hint_to_obs(hint)
                self.obs_cache[self.hint_ctr] = obs
                return obs
        else:
            return self.def_obs

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
        self.hint_ctr = 0
        self.reread_ctr = 0
        return self._observe()