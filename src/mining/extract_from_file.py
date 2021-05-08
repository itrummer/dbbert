'''
Created on May 8, 2021

@author: immanueltrummer

Extracts text from disk file (supports text and HTML files).
'''
from mining.web_util import extract_text
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='Extracts text from files')
parser.add_argument('input_path', type=str, help='Specify path to input file')
parser.add_argument('output_path', type=str, help='Specify path for output file')
args = parser.parse_args()
in_path = args.input_path
out_path = args.output_path

with open(in_path) as file:
    raw = file.read()
    # Distinguish file type
    if in_path.endswith('.html'):
        lines = extract_text(raw)
    else:
        lines = raw.split('.')
    rows = [(1, l) for l in lines]
    df = pd.DataFrame(rows, columns=['filenr', 'sentence'])
    df.to_csv(out_path, index=False)