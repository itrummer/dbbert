'''
Created on Nov 17, 2020

@author: immanueltrummer
'''  
import configparser
import PyPDF2
import re

# reading parameters from configuration file
print("Reading default MySQL configuration file")
config = configparser.ConfigParser()
config.read('configs/all_tunable_config.cnf')
p_to_text = {p:[] for p in config['mysqld-5.7']}

# reading parameter descriptions from manual
print("Reading text from PDF database manual")
pdfFileObj = open('manuals/msql5.7.pdf', 'rb')  
pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
print(pdfReader.numPages)

# collect sentences mentioning two parameters
multi_param_s = {}

# extract sentences that relate to parameters
#for page_nr in range(665,814):
for page_nr in range(0, pdfReader.numPages):
    if page_nr % 10 == 0:
        print(f"Scanning page {page_nr}")
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
with open('param_text.txt', 'w') as f:
    for p in p_to_text:
        if len(p_to_text[p])>0:
            f.write(f'{p}\t{". ".join(p_to_text[p])}\n')
        
# print sentences with multiple parameters
with open('multi_param.txt', 'w') as f:
    for s in multi_param_s:
        f.write(f'{multi_param_s[s]}\t{s}\n')
                
# close manual PDF object  
pdfFileObj.close()