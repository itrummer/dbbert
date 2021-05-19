'''
Created on Apr 22, 2021

@author: immanueltrummer
'''
from parameters.util import is_numerical, decompose_val
import unittest

class TestParameterExplorer(unittest.TestCase):
    """ Test parameter utility. """
    
    def test_is_numerical(self):
        """ Test identification of numerical parameters. """
        self.assertTrue(is_numerical('10MB'))
        self.assertTrue(is_numerical('10%'))
        self.assertTrue(is_numerical('10.0MB'))
        self.assertTrue(is_numerical('10'))
        self.assertTrue(is_numerical('0.000'))
        print (decompose_val('20and%'))