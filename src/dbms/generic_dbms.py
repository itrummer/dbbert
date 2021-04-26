'''
Created on Apr 2, 2021

@author: immanueltrummer
'''
from abc import ABC, abstractmethod
import copy

class ConfigurableDBMS(ABC):
    """ Represents a configurable database management system. """
    
    def __init__(self, db, user, password, unit_to_size):
        """ Initialize DB connection with given credentials. 
        
        Args:
            db: name of database to connect to
            user: name of database login
            password: password for database access
            unit_to_size: maps size units to byte size
        """
        self.db = db
        self.user = user
        self.password = password
        self.unit_to_size = unit_to_size
        self.config = {}
        self.connection = None
        self._connect()
        
    def __del__(self):
        """ Close DBMS connection if any. """
        self._disconnect()
    
    def can_query(self, sql):
        """ Returns True iff the query_one can be executed. """
        return True if self.query_one(sql) else False
    
    def can_set(self, param, value):
        """ Returns True iff we can set parameter to value. """
        current_value = self.get_value(param)
        # Try setting to new value
        try:
            valid = self.set_param_smart(param, value)
            self.set_param_smart(param, current_value)
            return valid
        except Exception:
            return False
    
    def changed(self):
        """ Return assignments for all changed parameters. """
        return copy.deepcopy(self.config)
    
    @abstractmethod
    def copy_db(self, source_db, target_db):
        """ Copy source to target database (overriding target). """
        pass
    
    @abstractmethod
    def exec_file(self, path):
        """ Executes all SQL queries in given file, returns error flag. """
        pass
    
    @abstractmethod
    def get_value(self, param):
        """ Returns current value for given parameter. """
        pass

    @abstractmethod
    def is_param(self, param):
        """ Returns True iff the given parameter can be configured. """
        pass

    @abstractmethod
    def query_one(self, sql):
        """ Runs SQL query_one and returns one result if query_one succeeds. """
        pass
    
    @abstractmethod
    def update(self, sql):
        """ Runs an SQL update and returns true iff the update succeeds. """
        pass

    @abstractmethod
    def reconfigure(self):
        """ Makes all parameter changes take effect (may require restart). 
        
        Returns:
            Whether reconfiguration was successful
        """
        pass

    @abstractmethod
    def reset_config(self):
        """ Reset all parameters to default values. """
        pass
    
    @abstractmethod
    def set_param(self, param, value):
        """ Set parameter to scaled value (exactly). """
        pass

    def set_param_smart(self, param, value):
        """ Set parameter to value, using simple transformations. """
        trans_value = self._transform_val(value)
        #print(f'set_param_smart: Trying to set {param} to {trans_value}')
        success = self.set_param(param, trans_value)
        #print(f'set_param_smart: {success}')
        return success
                
    @abstractmethod    
    def _connect(self):
        """ Establish connection to database, returns success flag. """
        pass
        
    @abstractmethod
    def _disconnect(self):
        """ Disconnect from database. """
        pass
            
    def _transform_val(self, value: str):
        """ Transforms parameter values using heuristic. """
        value = str(value)
        for unit in self.unit_to_size:
            size = self.unit_to_size[unit]
            value = value.replace(unit, size)
        return value