'''
Created on Apr 2, 2021

@author: immanueltrummer
'''
import copy
import os
import psycopg2
import time

from dbms.generic_dbms import ConfigurableDBMS

class PgConfig(ConfigurableDBMS):
    """ Reconfigurable Postgres DBMS instance. """
    
    def __init__(self, db, user):
        """ Initialize DB connection with given credentials. """
        self.db = db
        self.user = user
        self.config = {}
        self._connect()
        
    def __del__(self):
        """ Close DBMS connection if any. """
        self._disconnect()
            
    def _connect(self):
        """ Establish connection to database. """
        print(f'Trying to connect to {self.db} with user {self.user}')
        self.connection = psycopg2.connect(database = self.db, user = self.user, host = "localhost")
        
    def _disconnect(self):
        """ Disconnect from database. """
        if self.connection:
            print('Disconnecting ...')
            self.connection.close()
            
    def query(self, sql):
        """ Executes query and returns first result table cell or None. """
        try:
            self.connection.autocommit = True
            cursor = self.connection.cursor()
            cursor.execute(sql)
            return cursor.fetchone()[0]
        except Exception as e:
            print(e)
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
        query = f'alter system set {param} to {scaled_value}'
        success = self.update(query)
        if success:
            self.config[param] = scaled_value
            return True
        else:
            query = f'alter system set {param} to \'{scaled_value}\''
            return self.update(query)
    
    def reset_config(self):
        """ Reset all parameters to default values. """
        self.update('alter system reset all')
        self.config = {}
    
    def reconfigure(self):
        """ Makes parameter settings take effect. """
        self._disconnect()
        # TODO: this should not be hardcoded
        os.system('/opt/homebrew/bin/brew services restart postgresql')
        time.sleep(3)
        #os.system(r'/opt/homebrew/bin/pg_ctl -D /opt/homebrew/var/postgres restart')
        self._connect()
        
    def get_config(self):
        """ Return assignments for all parameters. """
        #return copy.deepcopy(self.config)
        return self.config