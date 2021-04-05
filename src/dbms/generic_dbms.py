'''
Created on Apr 2, 2021

@author: immanueltrummer
'''
from abc import ABC, abstractmethod
import re

class ConfigurableDBMS(ABC):
    """ Represents a configurable database management system. """
    
    @abstractmethod
    def query(self, sql):
        """ Runs SQL query and returns result if query succeeds. """
        pass
    
    @abstractmethod
    def update(self, sql):
        """ Runs an SQL update and returns true iff the update succeeds. """
        pass
    
    def can_query(self, sql):
        """ Returns True iff the query can be executed. """
        return True if self.query(sql) else False
    
    @abstractmethod
    def is_param(self, param):
        """ Returns True iff the given parameter can be configured. """
        pass
    
    def _scale(self, value, factor):
        """ Scales given parameter value. """
        # Return original value for scaling factor 1
        if factor == 1:
            return value
        # Is it numerical parameter, followed by unit?
        if re.match(r'\d+[a-zA-Z]*', value):
            # Separate number from unit, scale number
            digits = re.sub(r'(\d+)([a-zA-Z]*)', r'\g<1>', value)
            units = re.sub(r'(\d+)([a-zA-Z]*)', r'\g<1>', value)
            scaled = digits * factor
            return scaled + units
        else:
            # No scaling possible
            return value
    
    @abstractmethod
    def can_set(self, param, value, scale):
        """ Returns True iff given parameter can be set to scaled value. """
        pass
    
    @abstractmethod
    def get_value(self, param):
        """ Returns current value for given parameter. """
        pass
    
    @abstractmethod
    def set_param(self, param, value, scale):
        """ Set parameter to scaled value. """
        pass
    
    @abstractmethod
    def reset_config(self):
        """ Reset all parameters to default values. """
        pass
    
    @abstractmethod
    def reconfigure(self):
        """ Makes all parameter changes take effect (may require restart). """
        pass