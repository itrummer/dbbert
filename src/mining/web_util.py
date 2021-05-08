'''
Created on May 8, 2021

@author: immanueltrummer

Methods for extracting text from HTML sites.
'''
from bs4 import BeautifulSoup

def extract_text(html_src):
    """ Extract text passages from given URL body. 
    
    Args:
        html_src: HTML source code for text extraction.
        
    Returns:
        Lines from Web site or None if not retrievable.
    """
    try:
        parsed = BeautifulSoup(html_src, features="html.parser")
        for script in parsed(["script", "style"]):
            script.extract()
        text = parsed.get_text()
        print(f'Parsed text with length {len(text)}.')
        lines = [line.strip() for line in text.splitlines()]
        clean_lines = []
        for line in lines:
            clean_lines += [part.strip() for part in line.split("  ")]
        clean_lines = [line for line in clean_lines if len(line)>2]
        print(f'Extracted {len(clean_lines)} lines')
        return clean_lines
    except:
        return []