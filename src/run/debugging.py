'''
Created on Apr 17, 2021

@author: immanueltrummer

This module is for debugging.
'''
from dbms.postgres import PgConfig

dbms = PgConfig(db='tpch', user='immanueltrummer')
success = dbms.can_set('shared_buffers', '25%', 1)
print(success)