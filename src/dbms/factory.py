'''
Created on May 12, 2021

@author: immanueltrummer
'''
from dbms.postgres import PgConfig
from dbms.mysql import MySQLconfig

def from_file(config):
    """ Initialize DBMS object from configuration file. 
    
    Args:
        config: parsed configuration file
        
    Return:
        Object representing Postgres or MySQL
    """
    dbms_name = config['DATABASE']['dbms']
    if dbms_name == 'pg':
        return PgConfig.from_file(config)
    else:
        return MySQLconfig.from_file(config)