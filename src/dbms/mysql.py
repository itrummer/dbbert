'''
Created on Apr 7, 2021

@author: immanueltrummer
'''
from dbms.generic_dbms import ConfigurableDBMS

import mysql.connector
import os

class MySQLconfig(ConfigurableDBMS):
    """ Represents configurable MySQL database. """
    
    def __init__(self, db, user, password, bin_dir):
        """ Initialize DB connection with given credentials. 
        
        Args:
            db: name of MySQL database
            user: name of MySQL user
            password: password for database
            bin_dir: directory containing MySQL binaries (no trailing slash)
        """
        unit_to_size={'KB':'000', 'MB':'000000', 'GB':'000000000',
                      'K':'000', 'M':'000000', 'G':'000000000'}
        super().__init__(db, user, password, unit_to_size)
        self.bin_dir = bin_dir
        
    def __del__(self):
        """ Close DBMS connection if any. """
        super().__del__()
        
    def copy_db(self, source_db, target_db):
        """ Copy source to target database. """
        ms_clc_prefix = f'{self.bin_dir}/mysql -u{self.user} -p{self.password} '
        ms_dump_prefix = f'{self.bin_dir}/mysqldump -u{self.user} -p{self.password} '
        os.system(ms_dump_prefix + f' {source_db} > copy_db_dump')
        print('Dumped old database')
        os.system(ms_clc_prefix + f" -e 'drop database if exists {target_db}'")
        print('Dropped old database')
        os.system(ms_clc_prefix + f" -e 'create database {target_db}'")
        print('Created new database')
        os.system(ms_clc_prefix + f" {target_db} < copy_db_dump")
        print('Initialized new database')
            
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
    
    def query_one(self, sql):
        """ Runs SQL query_one and returns one result if it succeeds. """
        try:
            cursor = self.connection.cursor(buffered=True)
            cursor.execute(sql)
            return cursor.fetchone()[0]
        except Exception:
            return None
    
    def exec_file(self, path):
        """ Executes all SQL queries in given file and returns error flag. """
        error = True
        try:
            with open(path, 'r') as file:
                queries = file.read().split(';')
                for query in queries:
                    self.query_one(query)
            error = False
        except Exception as e:
            print(f'Exception: {e}')
        return error
    
    def update(self, sql):
        """ Runs an SQL update and returns true iff the update succeeds. """
        #print(f'Trying update {sql}')
        self.connection.autocommit = True
        cursor = self.connection.cursor(buffered=True)
        try:
            cursor.execute(sql)
            success = True
        except Exception as e:
            #print(f'Exception during update: {e}')
            success = False
        cursor.close()
        return success
    
    def is_param(self, param):
        """ Returns True iff the given parameter can be configured. """
        return self.can_query(f'show variables like \'{param}\'')
    
    def get_value(self, param):
        """ Returns current value for given parameter. """
        return self.query_one(f'select @@{param}')
    
    def set_param(self, param, value):
        """ Set parameter to given value. """
        #print(f'set_param: {param} to {value}')
        self.config[param] = value
        return self.update(f'set global {param}={value}')
    
    def all_params(self):
        """ Returns list of tuples, containing configuration parameters and values. """
        cursor = self.connection.cursor(buffered=True)
        cursor.execute('show global variables where variable_name != \'keyring_file_data\'')
        var_vals = cursor.fetchall()
        cursor.close()
        return var_vals
    
    def reset_config(self):
        """ Reset all parameters to default values. """
        var_vals = self.all_params()
        for var_val in var_vals:
            var, _ = var_val
            self.set_param(var, 'default')
        self.config = {}
    
    def reconfigure(self):
        """ Makes all parameter changes take effect (may require restart). 
        
        Returns:
            Whether reconfiguration was successful
        """
        # Currently, we consider no MySQL parameters requiring restart
        return True