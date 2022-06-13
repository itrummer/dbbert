'''
Created on Nov 15, 2021

@author: immanueltrummer
'''
import benchmark
import collections
import dataclasses
import dbms
import doc
import enum
import gym.spaces
import models.util
import numpy as np
import pandas as pd
import search.search_with_hints
import transformers
import typing

class DecisionType(enum.IntEnum):
    """ Describes next decision to make by agent. """
    PICK_FACTOR=0, # Pick a factor to multiply parameter value with
    PICK_WEIGHT=1, # Pick importance of tuning hint

class HintOrder(enum.IntEnum):
    """ The order in which tuning hints are considered. """
    DOCUMENT=0, # process hints in document order
    BY_PARAMETER=1, # prioritize hints about frequently mentioned parameters
    BY_STRIDE=2, # round robin over hint batches associated with parameters

def parse_order(config):
    """ Parse hint order from configuration file.
    
    Args:
        config: object represents configuration file
    
    Returns:
        order in which hints are considered
    """
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

@dataclasses.dataclass(frozen=True)
class LabeledObservation():
    """ Represents observation and associated logging text. """
    obs: typing.Any
    decision_txt: str
    choices: typing.List[str]
    scores: typing.List[float]
    
    def output(self,):
        """ Generates output describing observation. """
        print(f'Decision: {self.decision_txt}')
        print(f'Choices: {self.choices}')
        print(f'Scores: {self.scores}')


class NlpTuningEnv(gym.Env):
    """ Models database tuning using natural language hints
        as sequential decisions. Observations provided to the
        agent are derived from zero-shot classification results.
    """
    
    def __init__(
            self, docs: doc.collection.DocCollection, max_length, 
            hint_order, dbms: dbms.generic_dbms.ConfigurableDBMS, 
            benchmark: benchmark.evaluate.Benchmark, hardware, 
            hints_per_episode, nr_evals, scale_perf, scale_asg, objective):
        """ Initialize from given tuning documents, database, and benchmark. 
        
        Args:
            docs: collection of text documents with tuning hints
            max_length: maximum number of tokens per snippet
            hint_order: process tuning hints in this order
            dbms: database management system to tune
            benchmark: benchmark for which to tune system
            hardware: dictionary with entries for memory/disk/cores
            hints_per_episode: candidate hints before episode ends
            nr_evals: how many evaluations with extracted hints
            scale_perf: scale performance reward by this factor
            scale_asg: scale reward for successful assignments
            objective: describes the optimization goal
        """
        self.docs = docs
        self.max_length = max_length
        self.hints = self._ordered_hints(hint_order)
        self.dbms = dbms
        self.benchmark = benchmark
        self.hardware = hardware
        self.hints_per_episode = hints_per_episode
        self.nr_evals = nr_evals
        self.scale_perf = scale_perf
        self.scale_asg = scale_asg
        device = models.util.torch_device()
        self.bart = transformers.pipeline(
            'zero-shot-classification', 
            model='facebook/bart-large-mnli',
            device=device)
        self.explorer = search.search_with_hints.ParameterExplorer(
            dbms, benchmark, objective)
        self.decision = DecisionType.PICK_FACTOR
        self.factors = [0.25, 0.5, 1, 2, 4]
        self.weights = [1, 2, 4, 8, 16]
        self.action_space = gym.spaces.Discrete(5)
        self.obs_cache = {}
        self.observation_space = gym.spaces.Box(0, 1, (8,), np.float32)
        self.hint_ctr = 0
        self.episode_hint_ctr = 0
        self.nr_hints = len(self.hints)
        self.hint_to_weight = collections.defaultdict(lambda: 0)
        self.log = []
        self.log_dict = {}
        print('All hints considered for multi-doc tuning:')
        for i, (_, hint) in enumerate(self.hints):
            print(f'Hint {i}: {hint.param.group()} -> {hint.value.group()}')
    
    def reset(self):
        """ Initializes for new tuning episode. 
        
        Returns:
            current observations
        """
        self.decision = DecisionType.PICK_FACTOR
        self.base = None
        self.factor = None
        self.benchmark.print_stats()
        obs = self._observe()
        return obs
        
    def step(self, action):
        """ Performs one step in the environment. 
        
        Args:
            action: selected action
        
        Returns:
            observation, reward, termination flag, debugging info
        """
        print(f'Choice: {action}')
        reward = self._take_action(action)
        done = self._next_state()
        print(f'Done flag: {done}')
        if done:
            p_reward = self._finalize_episode()
            reward += p_reward
            self.log_dict['P-Reward'] = p_reward
        
        if self.log_dict:
            log_entry = pd.DataFrame([self.log_dict])
            self.log += [log_entry]
            self.log_dict = {}
        
        obs = self._observe()
        return obs, reward, done, {}
    
    def _finalize_episode(self):
        """ Return optimal benchmark reward when using weighted hints.
        
        Returns:
            reward representing improvements over default configuration
        """
        print('Finalizing episode!')
        if self.hint_to_weight:
            reward, config = self.explorer.explore(
                self.hint_to_weight, self.nr_evals)
            print(f'Achieved unscaled reward of {reward} using {config}.')
            self.hint_to_weight = collections.defaultdict(lambda: 0)
            return reward * self.scale_perf
        else:
            return 0.0

    def _hints_by_doc(self):
        """ Returns hints in document collection order. 
        
        Returns:
            list of pairs: document ID and hint.
        """
        hints = []
        for doc_id in range(self.docs.nr_docs):
            hints += [(doc_id, hint) for hint in self.docs.get_hints(doc_id)]
        return hints
        
    def _hints_by_param(self):
        """ Order hints by occurrence frequency of associated parameter.
        
        Returns:
            pairs (document ID and hint), ordered by # parameter mentions
        """
        ordered_hints = []
        for param, _ in self.docs.param_counts.most_common():
            param_hints = self.docs.param_to_hints[param]
            ordered_hints += param_hints
        return ordered_hints

    def _hints_by_stride(self):
        """ Round robin between parameters based on occurrence frequency.
        
        Returns:
            pairs (document ID and hint), balancing # mentions with diversity
        """
        ordered_hints = []
        hints_per_param = max([len(h) for h in self.docs.param_to_hints.values()])
        param_to_list = {p:list(v) for p, v in self.docs.param_to_hints.items()}
        step = 10
        for lb in range(0, hints_per_param, step):
            for param, _ in self.docs.param_counts.most_common():
                param_hints = param_to_list[param]
                nr_param_hints = len(param_hints)
                if lb < nr_param_hints:
                    ub = min(lb + step, nr_param_hints)
                    stride = param_hints[lb:ub]
                    ordered_hints += stride
        return ordered_hints

    def _next_state(self):
        """ Advance to next state in MDP and return termination flag. 
        
        Returns:
            flag indicating termination of episode
        """
        # Update decision and decide whether to advance
        if self.decision == DecisionType.PICK_FACTOR:
            self.decision = DecisionType.PICK_WEIGHT
        else:
            self.decision = DecisionType.PICK_FACTOR
            # Update hint counter
            self.hint_ctr += 1
            print(f'Hint counter: {self.hint_ctr}')
            if self.hint_ctr >= self.nr_hints:
                self.hint_ctr = 0
            # Update episode hint counter
            print(f'Episode hint counter before: {self.episode_hint_ctr}')
            self.episode_hint_ctr += 1
            print(f'Episode hint counter: {self.episode_hint_ctr}')
            print(f'Hints per episode: {self.hints_per_episode}')
            if self.episode_hint_ctr >= self.hints_per_episode:
                self.episode_hint_ctr = 0
                print('Episode finished!')
                return True
        return False
        
    def _observe(self):
        """ Generate observation for current decision and hint.
        
        Returns:
            Vector of floats: document, hint, decision, and BART scores.
        """
        obs_idx = (self.hint_ctr, int(self.decision))
        if obs_idx in self.obs_cache:
            l_obs = self.obs_cache[obs_idx]
        else:
            print(f'No warmup - hint counter: {self.hint_ctr}')
            _, hint = self.hints[self.hint_ctr]
            if self.decision == DecisionType.PICK_FACTOR:
                decision_txt = f'Deciding adaption of {hint}'
                choices = ['Decrease recommendation strongly', 
                           'Decrease recommendation', 
                           'Use recommendation', 
                           'Increase recommendation', 
                           'Increase recommendation strongly']
            else:
                decision_txt = f'Deciding weight of {hint}'
                v_weights = ['not', 'somewhat', 'quite', 'very', 'super']
                choices = [f'This hint is {w} important.' for w in v_weights]
            
            result = self.bart(hint.recommendation, choices)
            print(result)
            scores = []
            for choice in choices:
                choice_idx = result['labels'].index(choice)
                score = result['scores'][choice_idx]
                scores += [score]
            
            scaled_doc_id = hint.doc_id / self.docs.nr_docs
            scaled_hint_ctr = self.hint_ctr / self.nr_hints
            scaled_decision = float(self.decision) / 3
            scaled_vals = [scaled_doc_id, scaled_hint_ctr, scaled_decision]
            obs = scaled_vals + scores
            l_obs = LabeledObservation(obs, decision_txt, choices, scores)
        
        l_obs.output()
        self.obs_cache[obs_idx] = l_obs
        return l_obs.obs
    
    def _ordered_hints(self, hint_order):
        """ Returns hints according to specified order.
        
        Args:
            hint_order: specifies how to order hints
        
        Returns:
            pairs (document ID and hint), ordered as specified
        """
        if hint_order == HintOrder.BY_PARAMETER:
            return self._hints_by_param()
        elif hint_order == HintOrder.BY_STRIDE:
            return self._hints_by_stride()
        else:
            return self._hints_by_doc()

    def _process_hint(self, hint, action):
        """ Finishes processing current hint and returns direct reward.
        
        Args:
            hint: finish processing this hint
            action: last hint-related decision (hint weight)
        
        Returns:
            reward for DBMS accepting parameter value assignment
        """
        param = hint.param.group()
        value = str(int(self.base * self.factor)) + hint.val_unit
        success = self.dbms.can_set(param, value)
        assignment = (param, value)
        print(f'Trying assigning {param} to {value}')
        if success:
            weight = self.weights[action]
            self.hint_to_weight[assignment] += weight
            print(f'Adding assignment {assignment} with weight {weight}')
            print(f'Assignment {assignment} extracted from "{hint.passage}"')
            reward = 10 * self.scale_asg
        else:
            print(f'Assignment {assignment} was rejected')
            weight = -1
            reward = -10.0
        
        self.log_dict = {
            'Parameter':param, 'Recommendation':hint.recommendation, 
            'Inferred Type':self.type_text, 
            'Rec. Value':str(self.base) + ' ' + hint.val_unit, 
            'Factor':self.factor, 'Value':value, 'Weight':weight, 
            'Accepted':success, 'A-Reward':reward, 'P-Reward':0.0}
        return reward

    def _take_action(self, action):
        """ Process action and return obtained reward.
        
        Args:
            action: action (integer index)
        
        Returns:
            reward value for action
        """
        reward = 0
        _, hint = self.hints[self.hint_ctr]
        # Distinguish by decision type
        if self.decision == DecisionType.PICK_FACTOR:
            hint_type = hint.hint_type
            self.type_text = str(hint_type)
            if hint_type == doc.collection.HintType.DISK_RATIO:
                self.base = float(self.hardware['disk']) * hint.float_val
            elif hint_type == doc.collection.HintType.RAM_RATIO:
                self.base = float(self.hardware['memory']) * hint.float_val
            elif hint_type == doc.collection.HintType.CORES_RATIO:
                self.base = float(self.hardware['cores']) * hint.float_val
            elif hint_type == doc.collection.HintType.ABSOLUTE:
                self.base = hint.float_val
            else:
                raise ValueError(f'Unknown hint type: {hint_type}')
            self.factor = float(self.factors[action])
        else:
            reward = self._process_hint(hint, action)
        return reward