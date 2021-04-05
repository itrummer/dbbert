'''
Created on Apr 2, 2021

@author: immanueltrummer
'''
import psycopg2

from dbms.generic_dbms import ConfigurableDBMS

class PgConfig(ConfigurableDBMS):
    """ Reconfigurable Postgres DBMS instance. """
    
    def __init__(self, db='tpcc', user='immanueltrummer'):
        """ Initialize DB connection with given credentials. """
        self.db = db
        self.user = user
        self._connect()
        
    def __del__(self):
        """ Close DBMS connection if any. """
        if self.connection:
            self.connection.close()
            
    def _connect(self):
        """ Establish connection to database. """
        self.connection = psycopg2.connect(
            database = self.db, user = self.user, host = "localhost")
            
    def query(self, sql):
        """ Executes query and returns first result table cell or None. """
        try:
            self.connection.autocommit = True
            cursor = self.connection.cursor()
            cursor.execute(sql)
            return cursor.fetchone()[0]
        except Exception:
            return None
        
    def update(self, sql):
        """ Executes update and returns true iff the update succeeds. """
        try:
            self.connection.autocommit = True
            cursor = self.connection.cursor()
            cursor.execute(sql)
            return True
        except Exception:
            return False        
    
    def is_param(self, param):
        """ Returns True iff given parameter exists. """
        return self.can_query(f'show {param}')
    
    def can_set(self, param, value, factor):
        """ Returns True iff we can set parameter to scaled value. """
        current_value = self.get_value(param)
        # Try setting to new value
        try:
            valid = self.set_param(param, value, factor)
            self.set_param(param, current_value, 1)
            return valid
        except Exception:
            return False
        
    def get_value(self, param):
        """ Get current value of given parameter. """
        return self.query(f'show {param}')
        
    def set_param(self, param, value, factor):
        """ Set given parameter to scaled value. """
        scaled_value = self._scale(value, factor)
        query = f'set {param} to {scaled_value}'
        return self.update(query)
    
    def reset_config(self):
        """ Reset all parameters to default values. """
        self.connection.close()
        self._connect()
    
    def reconfigure(self):
        """ Makes parameter settings take effect. """
        # TODO: Still need to implement
        pass