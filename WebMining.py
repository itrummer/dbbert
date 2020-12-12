'''
Created on Dec 5, 2020

@author: immanueltrummer
'''
import configparser
import json
import time
from googleapiclient.discovery import build

# Configure access to Google engine


def google_query(query, api_key, cse_id, **kwargs):
    """ Retrieve results for given Web query """
    query_service = build("customsearch", "v1", 
                          developerKey=api_key)  
    query_results = query_service.cse().list(
        q=query, cx=cse_id, **kwargs).execute()
    return query_results['items']

# Read configuration parameters from file
config = configparser.ConfigParser()
with open('/home/ubuntu/liter2src/configs/pg_default.conf') as stream:
    config.read_string("[dummysection]\n" + stream.read())
    
# Create result object
all_param_text = json.loads('{}')
# Iterate over parameter sections
for section in config.sections():
    # Iterate over parameters in section
    for param in config[section]:
        time.sleep(1)
        print(f'Querying for parameter {param}')
        param_text = google_query(f"{param} faster", 
                    api_key, cse_id, num = 10)
        all_param_text[param] = param_text
        
# Write out result to file
with open('pg_web.txt', 'w') as file:
    json.dump(all_param_text, file)