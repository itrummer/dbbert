'''
Created on Apr 2, 2021

@author: immanueltrummer
'''
import os
import psycopg2
import time

from dbms.generic_dbms import ConfigurableDBMS

class PgConfig(ConfigurableDBMS):
    """ Reconfigurable Postgres DBMS instance. """
    
    def __init__(self, db, user, password=None):
        """ Initialize DB connection with given credentials. """
        super().__init__(db, user, password, {})
        
    def __del__(self):
        """ Close DBMS connection if any. """
        super().__del__()
            
    def _connect(self):
        """ Establish connection to database, returns success flag. """
        print(f'Trying to connect to {self.db} with user {self.user}')
        # Need to recover in case of bad configuration
        try:            
            self.connection = psycopg2.connect(
                database = self.db, user = self.user, 
                password = self.password, host = "localhost")
            return True
        except Exception:
            # Delete changes to default configuration and restart
            os.system('rm /opt/homebrew/var/postgres/postgresql.auto.conf')
            self.reconfigure()
            return False
        
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
        except Exception:
            return None
        
    def exec_file(self, path):
        """ Executes all SQL queries in given file. """
        error = True
        try:
            # TODO: this should not be hard-coded
            os.system(f'/opt/homebrew/bin/psql {self.db} -f {path} > query_results.txt')
            error = False
        except Exception as e:
            print(f'Exception: {e}')
        return error
        
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
        
    def get_value(self, param):
        """ Get current value of given parameter. """
        return self.query(f'show {param}')
        
    def set_param(self, param, value):
        """ Set given parameter to given value. """
        self.config[param] = value
        query = f'alter system set {param} to \'{value}\''
        return self.update(query)
    
    def reset_config(self):
        """ Reset all parameters to default values. """
        self.update('alter system reset all')
        self.config = {}
    
    def reconfigure(self):
        """ Makes parameter settings take effect. Returns true if successful. """
        self._disconnect()
        # TODO: this should not be hardcoded
        os.system('/opt/homebrew/bin/brew services restart postgresql')
        time.sleep(3)
        #os.system(r'/opt/homebrew/bin/pg_ctl -D /opt/homebrew/var/postgres restart')
        success = self._connect()
        return success