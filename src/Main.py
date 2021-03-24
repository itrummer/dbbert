'''
Created on Sep 6, 2020

@author: immanueltrummer
'''
""" Optimizes database configuration using NLP. """

import Configurations
import Gibbs
import Greedy

# Prepare tuning and perform NLP analysis
print("About to create tuning config")
tuning_config = Configurations.TuningConfig(
    '/etc/postgresql/10/main/postgresql.conf')
#tuning_config = Configurations.TuningConfig('scaled.conf')

# Generate interesting configurations
print("About to start Gibbs sampling")
samples = Gibbs.get_samples(tuning_config)

# Select most interesting conigurations to try
print("About to select samples greedily")
selected = Greedy.select(samples, 10)

print(selected)

# Try out most interesting configurations
print("About to try out selected samples")
# Evaluate selected configurations
evaluated = {}
for ctr, s in enumerate(selected):
    tuning_config.load_factors(s)
    e = tuning_config.evaluateConfig(
        'tpchs1', 'postgres', 'postgres')
    tuning_config.write_config(f'test_conf{ctr}')
    print(f'Configuration {ctr} took {e} ms')
    evaluated[e] = s
# Select minimum time configuration
opt_conf = evaluated[min(evaluated.keys())]
print(f'Best configuration: {opt_conf}')