'''
Created on Apr 2, 2021

@author: immanueltrummer
'''
from dbms.generic_dbms import ConfigurableDBMS
import os
import psycopg2
import time

class PgConfig(ConfigurableDBMS):
    """ Reconfigurable Postgres DBMS instance. """
    
    def __init__(self, db, user, password=None, 
                 restart_cmd="", data_dir="", timeout_s):
        """ Initialize DB connection with given credentials. 
        
        Args:
            db: name of Postgres database
            user: name of database user
            password: database password
            restart_cmd: command for restarting server
            data_dir: Postgres data directory
            timeout_s: per-query timeout in seconds
        """
        self.data_dir = data_dir
        unit_to_size={'KB':'kB', 'MB':'000kB', 'GB':'000000kB',
                      'K':'kB', 'M':'000kB', 'G':'000000kB'}
        super().__init__(db, user, password, 
                         unit_to_size, restart_cmd, timeout_s)
        
    @classmethod
    def from_file(cls, config):
        """ Initializes PgConfig object from configuration file. 
        
        Args:
            cls: class (currently, only PgConfig)
            config: configuration read from file
            
        Returns:
            new Postgres DBMS object
        """
        db_user = config['DATABASE']['user']
        db_name = config['DATABASE']['name']
        password = config['DATABASE']['password']
        restart_cmd = config['DATABASE']['restart_cmd']
        path_to_data = config['DATABASE']['data_dir']
        timeout_s = config['LEARNING']['timeout_s']
        return cls(db_name, db_user, password, 
                   restart_cmd, path_to_data, timeout_s)
        
    def __del__(self):
        """ Close DBMS connection if any. """
        super().__del__()
        
    def copy_db(self, source_db, target_db):
        """ Copy source to target database. """
        self.update(f'drop database if exists {target_db}')
        self.update(f'create database {target_db} with template {source_db}')
            
    def _connect(self):
        """ Establish connection to database, returns success flag. """
        print(f'Trying to connect to {self.db} with user {self.user}')
        # Need to recover in case of bad configuration
        try:            
            self.connection = psycopg2.connect(
                database = self.db, user = self.user, 
                password = self.password, host = "localhost")
            self.set_timeout(self.timeout_s)
            return True
        except Exception as e:
            # Delete changes to default configuration and restart
            print(f'Exception while trying to connect: {e}')
            #/opt/homebrew/var/postgres
            os.system(f'sudo rm {self.data_dir}/postgresql.auto.conf')
            self.reconfigure()
            return False
        
    def _disconnect(self):
        """ Disconnect from database. """
        if self.connection:
            print('Disconnecting ...')
            self.connection.close()

    def all_params(self):
        """ Return names of all tuning parameters. """
        cursor = self.connection.cursor()
        cursor.execute('select name from pg_settings where name != \'port\'')
        var_vals = cursor.fetchall()
        cursor.close()
        return [v[0] for v in var_vals]
                        
    def query_one(self, sql):
        """ Executes query_one and returns first result table cell or None. """
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
        
    def get_value(self, param):
        """ Get current value of given parameter. """
        return self.query_one(f'show {param}')
        
    def set_param(self, param, value):
        """ Set given parameter to given value. """
        self.config[param] = value
        query_one = f'alter system set {param} to \'{value}\''
        return self.update(query_one)
    
    def set_timeout(self, timeout_s):
        """ Set per-query timeout. """
        self.update(f"set statement_timeout = '{timeout_s}s'")

    def reset_config(self):
        """ Reset all parameters to default values. """
        self.update('alter system reset all')
        self.config = {}
    
    def reconfigure(self):
        """ Makes parameter settings take effect. Returns true if successful. """
        self._disconnect()
        os.system(self.restart_cmd)
        time.sleep(3)
        #/opt/homebrew/bin/brew services restart postgresql
        #os.system(r'/opt/homebrew/bin/pg_ctl -D /opt/homebrew/var/postgres restart')
        success = self._connect()
        return success
    