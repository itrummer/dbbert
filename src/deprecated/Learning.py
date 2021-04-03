'''
Created on Sep 6, 2020

@author: immanueltrummer
'''
import sys
sys.path.append(".")

import gym
import gym.spaces
import math
import numpy as np
import random
from configs import TuningConfig
#from bert_embedding import BertEmbedding
from sentence_transformers import SentenceTransformer

class ConfigEnv(gym.Env):
    """ Environment for learning to configure PG while 
        using NLP to understand parameter semantics
        based on parameter names.
    """
    def __init__(self, tuningConfig):
        """ Initializes action and observation spaces """
        self.tuning_config = tuningConfig
        # Initialize NLP via BERT
        #self.embedding = BertEmbedding()
        self.model = SentenceTransformer('bert-base-nli-mean-tokens')
        # Actions: decrease and test, keep, increase and test
        self.action_space = gym.spaces.Discrete(3)
        # Observation space relates to NLP
        self.observation_space = gym.spaces.Box(-10, 10, 
                shape=(768,), dtype=np.float32)
        # Get baseline time
        tuningConfig.restore_defaults()
        self.base_time = tuningConfig.evaluateConfig(
                'tpchs1', 'postgres', 'postgres')
        print(f'Base time is {self.base_time} ms')
        # Initialize current configuration file line
        self.reset()

    def make_observations(self, param):
        """ analyze description of parameter """
        #print(param.embedding.shape)
        return param.embedding

    def step(self, action):
        """ update random parameter """
        # Scale parameter as specified
        factor = None
        if action == 0:
            factor = 0.2
        elif action == 1:
            factor = 1
        else:
            factor = 5
        # Generate debugging output occasionally
        #if random.randrange(100)==0:
        #    print(f'Action selected was {action}')
        self.tuning_config.set_scale(self.lineID, factor)
        # Did we change the configuration?
        reward = None
        if factor != 1:
            new_time = self.tuning_config.evaluateConfig(
                    'tpchs1', 'postgres', 'postgres')
            reward = self.base_time - new_time - 60
            self.tuning_config.restore_defaults()
            # reward = -1
            # Generate debugging output
            if reward > 0:
                param = self.tuning_config.idToTunable[self.lineID]
                print(f'Reward {reward} for {param.tokens} with action {action}')
        else:
            reward = 0
        return np.zeros(shape=(768,)), reward, True, {}

    def reset(self):
        """ Resets configuration and selects random parameter """
        self.tuning_config.restore_defaults()
        tunableKeys = self.tuning_config.idToTunable.keys()
        self.lineID = random.choice(list(tunableKeys))
        param = self.tuning_config.idToTunable[self.lineID]
        return self.make_observations(param)

from keras import Input
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from rl.agents import SARSAAgent
from rl.policy import EpsGreedyQPolicy
import keras

# create prediction model for agent
def create_model():
    model = Sequential()
    model.add(Flatten(input_shape=(1, 768)))
    model.add(Dense(768, activation=keras.activations.sigmoid))
    model.add(Dense(768, activation=keras.activations.linear))
    model.add(Dense(3, activation=keras.activations.linear))
    return model

# create agent for configuration
def create_agent(env):
    model = create_model()
    print(model.summary())
    policy = EpsGreedyQPolicy()
    sarsa = SARSAAgent(model=model, nb_actions=env.action_space.n,
                     nb_steps_warmup=10, policy=policy)
    return sarsa

# Train agent for database tuning
print("About to create tuning config")
tuningConfig = TuningConfig('/etc/postgresql/10/main/postgresql.conf')
print("About to create environment")
env = ConfigEnv(tuningConfig)
print("About to create agent")
sarsa = create_agent(env)
print("Compiling SARSA agent")
sarsa.compile(Adam(lr=1e-4), metrics=['mae'])
print("About to train agent")
sarsa.fit(env, nb_steps=200000, visualize=False, verbose=1, log_interval=1000000)
print("Agent training is finished") 
