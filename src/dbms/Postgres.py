'''
Created on Mar 26, 2021

@author: immanueltrummer
'''
import psycopg2

class PgConfig:
    """ Functions for configuring Postgres. """

    def __init__(self):
        print('Initialization')
        self.create_conn()
        
    def create_conn(self):
        """ Initializes connection to database. """
        self.connection = psycopg2.connect(database = "tpcc", 
                user = "immanueltrummer", host = "localhost")
        
    def close_conn(self):
        """ Closes connection to database if any. """
        if self.connection:
            self.connection.close()
        
    def config(self, param, value):
        """ Sets configuration parameter to given value. """
        success = True
        with self.connection as connection:
            try:
                connection.autocommit = True
                cursor = connection.cursor()
                query = f'set {param} to {value};'
                cursor.execute(query)
            except Exception:
                success = False
        return success