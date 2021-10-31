'''
Created on Jun 24, 2021

@author: immanueltrummer
'''
from dbms import factory
from configparser import ConfigParser
from dbms.mysql import MySQLconfig

config = ConfigParser()
config.read('config/pg_tpch_base.ini')
pg = factory.from_file(config)
print(f'Tuning {len(pg.all_params())} parameters for Postgres')
ms = MySQLconfig('tpch', 'root', 'mysql1234-', '', '', 900)
print(f'Tuning {len(ms.all_params())} parameters for MySQL')
