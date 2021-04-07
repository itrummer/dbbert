'''
Created on Mar 22, 2021

@author: immanueltrummer
'''
import gym
import nlp.nlp_util as nlp
import numpy as np
from benchmark.evaluate import Benchmark
from collections import Counter
from gym.spaces import Box
from gym.spaces import Discrete
import torch
from random import randint
from doc.collection import DocCollection, TuningHint
from dbms.generic_dbms import ConfigurableDBMS

class TuningEnv(gym.Env):
    """ Trains agents in understanding tuning hints via NLP. """

    def __init__(self, docs: DocCollection, dbms: ConfigurableDBMS, benchmark: Benchmark):
        """ Initialize from given tuning documents, database, and benchmark. """
        self.docs = docs
        self.dbms = dbms
        self.benchmark = benchmark
        self.obs_cache = {}
        self.asg_count = Counter()
        _, self.def_millis = benchmark.evaluate(dbms)
        self.observation_space = Box(
            low=-10, high=10, shape=(1536,), dtype=np.float32)
        self.action_space = Discrete(2)
        self.reset()
        
    def step(self, action):
        """ Potentially apply hint and proceed to next one. """
        # Initialize return values
        reward = 1
        done = False
        # Check for end of episode
        if self.hint_idx >= self.nr_hints:
            # Run benchmark if new settings tried
            if self.dbms.changed():
                reward = self._eval_config()
            done = True
        # Execute action
        if not done and action == 1:
            hint: TuningHint = self.hints[self.hint_idx]
            reward, done = self._take_hint(hint)
        # Next step unless episode end
        if not done:
            self.hint_idx += 1
        return self._observe(), reward, done, {}
    
    def _eval_config(self):
        """ Benchmarks current DBMS configuration and returns reward. 
        
        Reconfigures DBMS with current configuration, then runs benchmark. 
        Next, it reconfigures to default configuration and runs same benchmark.
        
        Returns:
            reward value
        """
        # Benchmark current configuration, then reset
        can_reconfig = self.dbms.reconfigure()
        if can_reconfig:
            error, millis = self.benchmark.evaluate(self.dbms)
            self.dbms.reset_config()
            self.dbms.reconfigure()
            _, self.def_millis = self.benchmark.evaluate(self.dbms)
            # Reward is based on time improvements
            reward = self.def_millis - millis if not error else -10
        else:
            # Bad configuration prevents processing
            reward = -100
        return reward
    
    def _take_hint(self, hint):
        """ Change configuration as suggested in hint.
        
        Returns:
            reward: high if the configuration update was successful.
            done: true iff the update was not successful.
        """
        param = hint.param.group()
        value = hint.value.group()
        success = self.dbms.set_param_smart(param, value, 1)
        if success:
            output = f'Set {param} to {value}!'
            print(output)
            self.asg_count.update([(param, value)])
            reward = 5
            done = False
        else:
            reward = -10
            done = True
        return reward, done
        
    def _observe(self):
        """ Returns an observation. """
        if self.hint_idx >= self.nr_hints:
            return torch.zeros(1536)
        else:
            index = (self.doc_id, self.hint_idx)
            if index in self.obs_cache:
                return self.obs_cache[index]
            else:
                hint: TuningHint = self.hints[self.hint_idx]
                obs = self._hint_to_obs(hint)
                self.obs_cache[index] = obs
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
        obs = torch.zeros(1536)
        if not (obs_parts[0] is None or obs_parts[1] is None):
            obs = torch.cat((obs_parts[0], obs_parts[1]))
        return obs
        
    def reset(self):
        """ Initializes for new tuning document. """
        self.doc_id = randint(0, self.docs.nr_docs-1)
        print(f'Selected document nr. {self.doc_id}')
        self.hints = self.docs.get_hints(self.doc_id)
        self.nr_hints = len(self.hints)
        self.hint_idx = 0
        self.dbms.reset_config()
        #self.dbms.reconfigure()
        self.benchmark.print_stats()
        print(f'Default time: {self.def_millis}')
        print(self.asg_count)
        # for p_val in self.asg_count:
            # print(f'{p_val}\n')
        return self._observe()
