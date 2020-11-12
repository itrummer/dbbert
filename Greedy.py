'''
Created on Sep 6, 2020

@author: immanueltrummer
'''
""" Greedily select interesting configurations to try """

from scipy.spatial.distance import euclidean
import numpy as np

def error(targets, selected):
    """ Calculates error when approximating target
        by selected vectors.
    """
    error = 0
    for target in targets:
        min_dist =  np.inf # Depends on metric!
        for approx in selected:
            dist = euclidean(target, approx)
            #print(f'target: {target}')
            #print(f'approx: {approx}')
            #print(dist)
            min_dist = min(min_dist, dist)
        error += min_dist
    return error

def delta(targets, selected, added):
    """ Calculates delta (reduction in min-distance)
        when adding one more vector to approximate
        target.
    """
    expanded = selected.copy()
    expanded.append(added)
    prior_error = error(targets, selected)
    new_error = error(targets, expanded)
    delta = prior_error - new_error
    assert(delta>=0)
    return delta

def select(vectors, k):
    """ Select k most interesting vectors """
    selected = []
    for _ in range(k):
        max_delta = 0
        opt_vector = None
        for v in vectors:
            d = delta(vectors, selected, v)
            if d>=max_delta:
                max_delta = d
                opt_vector = v
        selected.append(opt_vector)
        print(f'Maximal delta: {max_delta}')
    return selected