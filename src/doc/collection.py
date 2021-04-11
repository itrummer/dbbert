'''
Created on Apr 3, 2021

@author: immanueltrummer
'''
import pandas as pd
import re
import nlp.nlp_util

class TuningHint():
    """ Represents a single tuning hint. """
    def __init__(self, doc_id, passage, param, value):
        """ Initializes tuning hint for given passage. 
        
        Args:
            doc_id: document from which hint was extracted
            passage: A text passage containing the hint.
            param: match object referencing parameter in passage.
            value: match object referencing value in passage.
        """
        self.doc_id = doc_id
        self.passage = passage
        self.param = param
        self.value = value
            
class DocCollection():
    """ Represents a collection of documents. """
    
    def __init__(self, docs_path, dbms=None):
        """ Reads tuning passages from a file. """
        self.docs = pd.read_csv(docs_path)
        self.docs.fillna('', inplace=True)
        if dbms:
            self.docs = self.docs[self.docs['dbms'] == dbms]
        self.nr_docs = self.docs['filenr'].max()
        self.nr_passages = []
        self.passages_by_doc = []
        for doc_id in range(self.nr_docs):
            passages = self._doc_passages(doc_id+1)
            self.passages_by_doc.append(passages)
            self.nr_passages.append(len(passages))
        # Prepare caching of tuning hints
        self.doc_to_hints = {} 
        # Output a summary of data read
        print('Sample of tuning hints:')
        print(self.docs.sample())
        print(f'Nr. documents read: {self.nr_docs}')
        print(f'Nr. passages by doc: {self.nr_passages}')

    def _doc_passages(self, doc_id):
        """ Extract text snippets from given document. """ 
        snippets_idx = self.docs['filenr'] == doc_id
        snippets = self.docs.loc[snippets_idx, 'sentence']
        # Join snippets into larger passages
        passages = []
        passage = []
        p_length = 0
        for snippet in snippets:
            s_length = nlp.nlp_util.tokenize(snippet)['input_ids'].shape[1]
            print(snippet)
            print(s_length)
            p_length += s_length
            if p_length > 512:
                # Start new passage
                passages.append(" ".join(passage))
                passage = [snippet]
                p_length = 0
            else:
                # Append snippet to passage
                passage.append(snippet)
                p_length += s_length
        return passages
    
    def get_hints(self, doc_id):
        """ Returns candidate tuning hints extracted from given document. 
        
        Returns:
            List of candidate tuning hints.
        """
        if doc_id in self.doc_to_hints:
            return self.doc_to_hints[doc_id]
        else:
            print(f'Creating hints for document {doc_id}')
            hints = []
            passages = self.passages_by_doc[doc_id]
            for passage in passages:
                params = re.finditer(r'[a-z_]+_[a-z]+', passage)
                #values = re.finditer(r'\d+[a-zA-Z]*|on|off', passage)
                values = re.finditer(r'\d+[a-zA-Z]*%{0,1}', passage)
                for param in params:
                    for value in values:
                        hint = TuningHint(doc_id, passage, param, value)
                        hints.append(hint)
            self.doc_to_hints[doc_id] = hints
            return hints