'''
Created on Apr 9, 2021

@author: immanueltrummer
'''
import argparse
import pandas as pd
from googleapiclient.discovery import build
from urllib.request import urlopen
from mining.web_util import extract_text

def google_query(query_one, api_key, cse_id):
    """ Uses specified search engine to query_one, returns results. 
    
    Returns:
        A list of search result items.
    """
    query_service = build(
        "customsearch", "v1", developerKey=api_key)
    all_results = []
    for start in range(1, 100, 10):
        print(f'Retrieving results starting from index {start}')
        query_results = query_service.cse().list(
            q=query_one, cx=cse_id, start=start, lr='lang_en', 
            dateRestrict='y1').execute()
        all_results += query_results['items']
    return all_results

def get_web_text(url):
    """ Extract text passages from given URL body. 
    
    Returns:
        Lines from Web site or None if not retrievable.
    """
    try:
        html_src = urlopen(url, timeout=5).read()
        print(f'Retrieved url {url}')
        return extract_text(html_src)
    except:
        return []

def can_parse(result):
    """ Returns true iff the search result can be used. """
    return True if not '.pdf' in result['link'] else False

# Parse command line arguments
parser = argparse.ArgumentParser(description='Retrieve results of Google query_one')
parser.add_argument('query_one', type=str, help='Specify Google search query_one')
parser.add_argument('key', type=str, help='Specify the Google API key')
parser.add_argument('cse', type=str, help='Specify SE ID (https://programmablesearchengine.google.com/)')
parser.add_argument('out_path', type=str, help='Specify path to output file')
args = parser.parse_args()
print(args)

# Write Google query_one results into file
items = google_query(args.query_one, args.key, args.cse)
rows = []
for docid, result in enumerate(items):
    url = result['link']
    print(url)
    if can_parse(result):
        print('Processing document')
        lines = get_web_text(url)
        for line in lines:
            rows.append([docid, line])
    else:
        print('Did not process document')
data = pd.DataFrame(rows, columns=['filenr', 'sentence'])
data.to_csv(args.out_path, index=False)