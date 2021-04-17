'''
Created on Apr 2, 2021

@author: immanueltrummer
'''
from abc import ABC, abstractmethod
import copy
import re

class ConfigurableDBMS(ABC):
    """ Represents a configurable database management system. """
    
    def __init__(self, db, user, password, main_memory, unit_to_size):
        """ Initialize DB connection with given credentials. 
        
        Args:
            db: name of database to connect to
            user: name of database login
            password: password for database access
            main_memory: main memory size in bytes
            unit_to_size: maps size units to byte size
        """
        self.db = db
        self.user = user
        self.password = password
        self.main_memory = main_memory
        self.unit_to_size = unit_to_size
        self.config = {}
        self.connection = None
        self._connect()
        
    def __del__(self):
        """ Close DBMS connection if any. """
        self._disconnect()
        
    @abstractmethod    
    def _connect(self):
        """ Establish connection to database, returns success flag. """
        pass
        
    @abstractmethod
    def _disconnect(self):
        """ Disconnect from database. """
        pass
    
    @abstractmethod
    def query(self, sql):
        """ Runs SQL query and returns result if query succeeds. """
        pass
    
    @abstractmethod
    def exec_file(self, path):
        """ Executes all SQL queries in given file, returns error flag. """
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
            
    def _transform_val(self, value: str):
        """ Transforms parameter values using heuristic. """
        if re.match('\d+%', value):
            # Assume percentage refers to main memory
            percentage = int(re.sub('(\d+)(%)', '\g<1>', value))
            memory = int(self.main_memory * percentage/100)
            return str(memory)
        else:
            for unit in self.unit_to_size:
                size = self.unit_to_size[unit]
                value = value.replace(unit, size)
            return value
      
    def can_set(self, param, value, factor):
        """ Returns True iff we can set parameter to scaled value. """
        current_value = self.get_value(param)
        # Try setting to new value
        try:
            valid = self.set_param_smart(param, value, factor)
            self.set_param_smart(param, current_value, 1)
            return valid
        except Exception:
            return False
    
    @abstractmethod
    def get_value(self, param):
        """ Returns current value for given parameter. """
        pass
    
    def set_param_smart(self, param, value, factor):
        """ Set parameter to scaled value, trying different versions if needed. """
        scaled_value = self._scale(value, factor)
        trans_value = self._transform_val(scaled_value)
        print(f'Trying to set {param} to {trans_value}')
        success = self.set_param(param, trans_value)
        if not success: 
            success = self.set_param(param, '\'' + trans_value + '\'')
        if success:
            self.config[param] = scaled_value
        return success
    
    @abstractmethod
    def set_param(self, param, value):
        """ Set parameter to scaled value (exactly). """
        pass
    
    @abstractmethod
    def reset_config(self):
        """ Reset all parameters to default values. """
        pass
    
    @abstractmethod
    def reconfigure(self):
        """ Makes all parameter changes take effect (may require restart). 
        
        Returns:
            Whether reconfiguration was successful
        """
        pass
    
    def changed(self):
        """ Return assignments for all changed parameters. """
        return copy.deepcopy(self.config)