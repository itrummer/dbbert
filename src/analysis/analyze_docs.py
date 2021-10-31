'''
Created on Jun 22, 2021

@author: immanueltrummer
'''
from dbms import factory
from configparser import ConfigParser
from doc.collection import DocCollection
from dbms.mysql import MySQLconfig

#,, ('ms', 'tuning_docs/mysql100')

config = ConfigParser()
config.read('config/pg_tpch_base.ini')
pg = factory.from_file(config)
ms = MySQLconfig('tpch', 'root', 'mysql1234-', '', '', 900)

for doc_id, doc_path, dbms in [('pg', 'tuning_docs/postgres100', pg),
                               ('ms', 'tuning_docs/mysql100', ms)]:
    docs = DocCollection(
        docs_path=doc_path, 
        dbms=dbms, size_threshold=128,
        use_implicit=1, filter_params=1)
    
    asg_cnt, param_cnt = docs._assignment_stats()
    print(asg_cnt)
    print(param_cnt)
    
    for c_id, counter in [('asg', asg_cnt), ('param', param_cnt)]:
        row = 0
        with open(f'analysis/{doc_id}_{c_id}', 'w') as file:
            for entity, cnt in counter.most_common():
                row += 1
                file.write(f'{row}\t{entity}\t{cnt}\n')