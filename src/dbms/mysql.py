'''
Created on Apr 7, 2021

@author: immanueltrummer
'''
from dbms.generic_dbms import ConfigurableDBMS

import mysql.connector

class MySQLconfig(ConfigurableDBMS):
    """ Represents configurable MySQL database. """
    
    def __init__(self, db, user, password):
        """ Initialize DB connection with given credentials. """
        super().__init__(db, user, password)
        
    def __del__(self):
        """ Close DBMS connection if any. """
        super().__del__()
            
    def _connect(self):
        """ Establish connection to database, returns success flag. """
        print(f'Trying to connect to {self.db} with user {self.user}')
        # Need to recover in case of bad configuration
        try:            
            self.connection = mysql.connector.connect(
                database=self.db, user=self.user, 
                password=self.password, host="localhost")
            return True
        except Exception:
            # TODO: how to recover for MySQL?
            return False
        
    def _disconnect(self):
        """ Disconnect from database. """
        if self.connection:
            print('Disconnecting ...')
            self.connection.close()
    
    def query(self, sql):
        """ Runs SQL query and returns result if query succeeds. """
        try:
            cursor = self.connection.cursor()
            cursor.execute(sql)
            return cursor.fetchone()[0]
        except Exception as e:
            print(e)
            return None
    
    def exec_file(self, path):
        """ Executes all SQL queries in given file and returns error flag. """
        error = True
        try:
            self.update(f'source \'{path}\'')
            error = False
        except Exception as e:
            print(f'Exception: {e}')
        return error
    
    def update(self, sql):
        """ Runs an SQL update and returns true iff the update succeeds. """
        try:
            self.connection.autocommit = True
            cursor = self.connection.cursor()
            cursor.execute(sql)
            cursor.commit()
            cursor.close()
            return True
        except Exception:
            return False
    
    def is_param(self, param):
        """ Returns True iff the given parameter can be configured. """
        return True if self.query(f'show variables like \'{param}\'') else False
    
    def get_value(self, param):
        """ Returns current value for given parameter. """
        return self.query(f'@@{param}')
    
    def set_param(self, param, value):
        """ Set parameter to given value. """
        return self.update(f'set global {param}={value}')
    
    def reset_config(self):
        """ Reset all parameters to default values. """
        for param, _ in self.changed():
            self.set_param(param, 'DEFAULT')
        self.config = {}
    
    def reconfigure(self):
        """ Makes all parameter changes take effect (may require restart). 
        
        Returns:
            Whether reconfiguration was successful
        """
        # Currently, we consider no MySQL parameters requiring restart
        pass