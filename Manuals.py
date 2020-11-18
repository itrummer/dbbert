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
pdfFileObj = open('manuals/refman-5.7-en.pdf', 'rb')  
pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
print(pdfReader.numPages)

# collect sentences mentioning two parameters
multi_param_s = {}

# extract sentences that relate to parameters
for page_nr in range(665,814):
#for page_nr in range(0, pdfReader.numPages):
    if page_nr % 10 == 0:
        print(f"Scanning page {page_nr}")
    page = pdfReader.getPage(page_nr)
    raw_text = page.extractText()
    text = raw_text.replace("\n", " ")
    all_sentences = re.split("\.", text)
    for s in all_sentences:
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
                        break
                if not substring:
                    p_mentions.append(p)
        # Check for multi-parameter mentions
        if len(p_mentions) > 1:
            multi_param_s[s] = p_mentions

# print parameters with associated sentences
for p in p_to_text:
    if len(p_to_text[p])>0:
        print(p)
        print(p_to_text[p])
        
# print sentences with multiple parameters
print(" *** MULTI-PARAMETER SENTENCES *** ")
for s in multi_param_s:
    print(" --- ")
    print(multi_param_s[s])
    print(s)
                
# close manual PDF object  
pdfFileObj.close()