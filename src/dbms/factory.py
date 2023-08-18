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


def from_args(args):
    """ Initialize DBMS object from command line arguments.
    
    Args:
        args: dictionary containing command line arguments.
    
    Returns:
        DBMS object.
    """
    if args.dbms == 'pg':
        return PgConfig(
            args.db_name, args.db_user, args.db_pwd, args.restart_cmd, 
            args.recover_cmd, args.timeout_s)
    elif args.dbms == 'ms':
        return MySQLconfig(
            args.db_name, args.db_user, args.db_pwd, args.restart_cmd, 
            args.recover_cmd, args.timeout_s)
    else:
        raise ValueError(f'DBMS {args.dbms} not supported!')