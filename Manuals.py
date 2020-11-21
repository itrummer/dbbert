'''
Created on Nov 17, 2020

@author: immanueltrummer
'''  
import configparser
import PyPDF2
import os
import re
import sys

# Read location of manual and configuration
conf_path = sys.argv[1]
print(f'Path to config file: {conf_path}')
man_path = sys.argv[2]
print(f'Path to manual .pdf: {man_path}')
out_dir = sys.argv[3]
print(f'Path to output directory: {out_dir}')
if not os.path.isdir(out_dir):
    print('Output directory does not exist!')
    sys.exit(1)

# Reading parameters from configuration file
print("Reading default configuration file")
config = configparser.ConfigParser()
with open(conf_path) as stream:
    config.read_string("[dummysection]\n" + 
                       stream.read())

# reading parameter descriptions from manual
print("Reading text from PDF database manual")
pdfFileObj = open(man_path, 'rb')  
pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
print(pdfReader.numPages)

# Extract parameters
all_params = []
for s in config.sections():
    print(f'Section {s}', flush=True)
    all_params += [p for p in config[s]]
p_to_text = {p:[] for p in all_params}
# Collect sentences mentioning two parameters
multi_param_s = {}

# Extract sentences that relate to parameters
nr_pages = pdfReader.numPages
for page_nr in range(0, nr_pages):
    if page_nr % 10 == 0:
        print(f"Scanning page {page_nr} of {nr_pages}")
    page = pdfReader.getPage(page_nr)
    raw_text = page.extractText()
    text = raw_text.replace("\n", " ")
    snippets = re.split("\.", text)
    """
    # Split input text into snippets
    words = re.split("\\n|\.| ", text)
    snippets = []
    snip_len = 20
    for i in range(0, len(words), snip_len):
        snippet = " ".join(words[i:min(i+snip_len, len(words))])
        snippets.append(snippet)
    """
    for s in snippets:
        # Truncate if required
        if len(s) > 250:
            s = s[0:250]
        # Iterate over parameters
        p_mentions = []
        for p in p_to_text:
            if p in s:
                p_to_text[p] += [s]
                # Avoid parameter substrings
                substring = False
                for old_p in p_mentions:
                    if p in old_p:
                        substring = True
                    elif old_p in p:
                        p_mentions.remove(old_p)
                if not substring:
                    p_mentions.append(p)
        # Check for multi-parameter mentions
        if len(p_mentions) > 1:
            multi_param_s[s] = p_mentions

# print parameters with associated sentences
with open(f'{out_dir}/param_text.txt', 'w') as f:
    for p in p_to_text:
        if len(p_to_text[p])>0:
            f.write(f'{p}\t{". ".join(p_to_text[p])}\n')
        
# print sentences with multiple parameters
with open(f'{out_dir}/multi_param.txt', 'w') as f:
    for s in multi_param_s:
        f.write(f'{multi_param_s[s]}\t{s}\n')
                
# close manual PDF object  
pdfFileObj.close()