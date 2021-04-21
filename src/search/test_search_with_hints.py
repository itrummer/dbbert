'''
Created on Apr 20, 2021

@author: immanueltrummer
'''
from search.search_with_hints import ParameterExplorer
import unittest

class TestParameterExplorer(unittest.TestCase):
    """ Test parameter explorer. """
    
    @classmethod
    def setUpClass(cls):
        cls.explorer = ParameterExplorer(None, None)
    
    def test_config_selection(self):
        """ Test selection of next configuration. """
        hint_to_weight = {}
        hint_to_weight[('innodb_buffer_pool_size', 2)] = 10
        hint_to_weight[('innodb_buffer_pool_size', 4)] = 10
        print(self.explorer._gather_values(hint_to_weight))
        print(self.explorer._select_configs(hint_to_weight, 2))