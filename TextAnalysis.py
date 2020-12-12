'''
Created on Dec 5, 2020

@author: immanueltrummer
'''
import json

with open('pg_web.txt') as file:
    web_results = json.load(file)
    
for param in web_results:
    print(f'{param}: {web_results[param][0]["snippet"]}')
    print()
    