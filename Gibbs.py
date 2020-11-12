'''
Created on Sep 6, 2020

@author: immanueltrummer
'''
from scipy.spatial.distance import cosine
import random

def calculate_distances(tuning_config):
    """ Calculate linguistic distance between
        pairs of configuration parameters. 
    """
    bert_dist = {}
    for line_1 in tuning_config.idToTunable:
        for line_2 in tuning_config.idToTunable:
            param1 = tuning_config.idToTunable[line_1]
            param2 = tuning_config.idToTunable[line_2]
            dist = cosine(param1.embedding, param2.embedding)
            bert_dist[(line_1, line_2)] = dist
    return bert_dist

def output_distances(tuning_config, bert_dist, path):
    """ Writes out parameter pairs with distances to file """
    # Sort parameter pairs by cosine distance (ascending)
    file = open(path, 'w')
    by_dist = {k : v for k, v in sorted(bert_dist.items(), 
                                        key = lambda item:item[1])}
    for (line_1, line_2), dist in by_dist.items():
        text_1 = tuning_config.idToLine[line_1]
        text_2 = tuning_config.idToLine[line_2]
        file.write(f'{text_1} to {text_2}: {dist}')
        file.write('\n')
    file.close()

def init_factors(tuning_config):
    """ Initialize parameter scaling factors randomly.
    """
    random.seed()
    for line in tuning_config.idToTunable:
        factor = random.choice([0.2, 1, 5])
        tuning_config.set_scale(line, factor)

def sample_factor(tuning_config, bert_dist, line):
    """ Draw sample factor, probability is based
        on assignments for other coniguration lines
        with similar semantics according to BERT. """
    # Init probabilities of decrease, same, increase
    probs = [1, 1, 1]
    for other_line in tuning_config.idToTunable:
        if other_line != line:
            factor = tuning_config.idToFactor[other_line]
            dist = bert_dist[(line, other_line)]
            #print(f'Other factor: {factor}')
            if factor < 1:
                probs[0] += 1.0/dist
            elif factor > 1:
                probs[2] += 1.0/dist
            else:
                probs[1] += 1.0/dist
    #print(probs)
    # Sample using probabilities
    return random.choices([0.2, 1, 5], weights=probs)[0]

def iterate(tuning_config, bert_dist):
    """ One iteration of Gibbs sampling: we assign
        new factor to each tuning parameter.
    """
    for line in tuning_config.idToTunable:
        factor = sample_factor(tuning_config, bert_dist, line)
        tuning_config.set_scale(line, factor)

def get_samples(tuning_config):
    """ Returns combinations of (relative) configuration
        parameter settings, obtained via BERT-based NLP
        and Gibbs sampling.
    """
    # Calculate linguistic distances for parameter pairs
    bert_dist = calculate_distances(tuning_config)
    # Initialize collection of configurations
    samples = []
    # Initialize scaling factors randomly
    init_factors(tuning_config)
    # Perform Gibbs sampling iterations
    warmup_iters = 200
    for i in range(400):
        # Output progress
        if i % 100 == 0:
            print(f'Starting iteration {i}')
        # Update parameters
        iterate(tuning_config, bert_dist)
        # Add to collection after warmup
        if i > warmup_iters:
            factors = tuning_config.get_factors()
            samples.append(factors)
            """
            fresh = True
            for prior in samples:
                if (prior==factors).all():
                    fresh = False
            if fresh:
                samples.append(factors)
            """
    return samples

# Write out final configuration
#tuning_config.writeConfig('GibbsFinal.txt')
# Summarize sampled factors
#print(samples)