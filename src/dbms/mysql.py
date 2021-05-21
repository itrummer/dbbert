'''
Created on Apr 7, 2021

@author: immanueltrummer
'''
from dbms.generic_dbms import ConfigurableDBMS

import mysql.connector
import os
import time

class MySQLconfig(ConfigurableDBMS):
    """ Represents configurable MySQL database. """
    
    def __init__(self, db, user, password, bin_dir, restart_cmd, timeout_s):
        """ Initialize DB connection with given credentials. 
        
        Args:
            db: name of MySQL database
            user: name of MySQL user
            password: password for database
            bin_dir: directory containing MySQL binaries (no trailing slash)
            timeout_s: per-query timeout in seconds
        """
        unit_to_size={'KB':'000', 'MB':'000000', 'GB':'000000000',
                      'K':'000', 'M':'000000', 'G':'000000000'}
        super().__init__(db, user, password, 
                         unit_to_size, restart_cmd, timeout_s)
        self.bin_dir = bin_dir
        self.global_vars = [t[0] for t in self.query_all(
            'show global variables where variable_name != \'keyring_file_data\'')]
        self.server_cost_params = [t[0] for t in self.query_all(
            'select cost_name from mysql.server_cost')]
        self.engine_cost_params = [t[0] for t in self.query_all(
            'select cost_name from mysql.engine_cost')]
        self.all_variables = self.global_vars + \
            self.server_cost_params + self.engine_cost_params
            
        print(f'Global variables: {self.global_vars}')
        print(f'Server cost parameters: {self.server_cost_params}')
        print(f'Engine cost parameters: {self.engine_cost_params}')
        print(f'All parameters: {self.all_variables}')
        
    @classmethod
    def from_file(cls, config):
        """ Initialize DBMS from configuration file. 
        
        Args:
            cls: class (currently: MySQLconfig only)
            config: configuration read from file
            
        Returns:
            new MySQL DBMS object
        """
        db_user = config['DATABASE']['user']
        db_name = config['DATABASE']['name']
        password = config['DATABASE']['password']
        bin_dir = config['DATABASE']['bin_dir']
        restart_cmd = config['DATABASE']['restart_cmd']
        timeout_s = config['LEARNING']['timeout_s']
        return cls(db_name, db_user, password, 
                   bin_dir, restart_cmd, timeout_s)
        
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
            self.set_timeout(self.timeout_s)
            return True
        except Exception as e:
            print(f'Exception while trying to connect to MySQL: {e}')
            return False
        
    def _disconnect(self):
        """ Disconnect from database. """
        if self.connection:
            print('Disconnecting ...')
            self.connection.close()
    
    def exec_file(self, path):
        """ Executes all SQL queries in given file and returns error flag. """
        try:
            self.connection.autocommit = True
            with open(path) as file:
                sql = file.read()
                for query in sql.split(';'):
                    cursor = self.connection.cursor()
                    cursor.execute(query)
                    cursor.close()
            error = False
        except Exception as e:
            error = True
            print(f'Exception execution {path}: {e}')
        return error
    
    def query_one(self, sql):
        """ Runs SQL query_one and returns one result if it succeeds. """
        try:
            cursor = self.connection.cursor(buffered=True)
            cursor.execute(sql)
            return cursor.fetchone()[0]
        except Exception:
            return None
    
    def query_all(self, sql):
        """ Runs SQL query and returns all results if it succeeds. """
        try:
            cursor = self.connection.cursor(buffered=True)
            cursor.execute(sql)
            results = cursor.fetchall()
            cursor.close()
            return results
        except Exception as e:
            print(f'Exception in mysql.query_all: {e}')
            return None
    
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
        return param in self.all_variables
    
    def get_value(self, param):
        """ Returns current value for given parameter. """
        if param in self.global_vars:
            return self.query_one(f'select @@{param}')
        elif param in self.server_cost_params:
            return self.query_one(
                "select case when cost_value is NULL then default_value " \
                "else cost_value end from mysql.server_cost " \
                f"where cost_name='{param}'")
        elif param in self.engine_cost_params:
            return self.query_one(
                "select case when cost_value is NULL then default_value " \
                "else cost_value end from mysql.engine_cost " \
                f"where cost_name='{param}'")
        else:
            return None
    
    def set_param(self, param, value):
        """ Set parameter to given value. """
        if param in self.global_vars:
            success = self.update(f'set global {param}={value}')
        elif param in self.server_cost_params:
            success = self.update(
                f"update mysql.server_cost set cost_value={value} where cost_name='{param}'")
        elif param in self.engine_cost_params:
            success = self.update(
                f"update mysql.engine_cost set cost_value={value} where cost_name='{param}'")
        else: 
            success = False
        if success:
            self.config[param] = value
        return success
    
    def set_timeout(self, timeout_s):
        """ Set per-query timeout. """
        timeout_ms = int(timeout_s * 1000)
        self.update(f"set session max_execution_time = {timeout_ms}")
    
    def all_params(self):
        """ Returns list of tuples, containing configuration parameters and values. """
        return self.all_variables
    
    def reset_config(self):
        """ Reset all parameters to default values. """
        self._disconnect()
        os.system(self.restart_cmd)
        time.sleep(2)
        self._connect()
        self.update('update mysql.server_cost set cost_value = NULL')
        self.update('update mysql.engine_cost set cost_value = NULL')
        self.config = {}
    
    def reconfigure(self):
        """ Makes all parameter changes take effect (may require restart). 
        
        Returns:
            Whether reconfiguration was successful
        """
        # Optimizer cost parameters requires flush and reconnect
        self.update('flush optimizer_costs')
        self._disconnect()
        self._connect()
        # Currently, we consider no MySQL parameters requiring restart
        return True