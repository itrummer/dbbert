'''
Created on Apr 3, 2021

@author: immanueltrummer
'''
from collections import Counter, defaultdict
from doc.util import get_parameters, get_values
from parameters.util import decompose_val
import pandas as pd
import re
import nlp.nlp_util
from dbms.generic_dbms import ConfigurableDBMS
from sentence_transformers import SentenceTransformer, util
from nltk.metrics.aline import similarity_matrix

class TuningHint():
    """ Represents a single tuning hint, assigning a parameter to a value. """
    
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
        self.float_val, self.val_unit = decompose_val(value.group())
            
class DocCollection():
    """ Represents a collection of documents with tuning hints. """
    
    def __init__(self, docs_path, dbms:ConfigurableDBMS, 
                 size_threshold, filter_params, consider_implicit):
        """ Reads tuning passages from a file. 
        
        Reads passages containing tuning hints from a text. Tries
        to filter passages to interesting ones. If given, a DBMS
        is used to filter to passages containing parameter names.
        
        Args:
            docs_path: path to document with tuning hints.
            dbms: database management system (optional).
            size_threshold: start new passage after so many tokens.
            filter_params: whether to filter hints by their parameters.
            consider_implicit: whether to consider implicit hints.
        """
        self.dbms = dbms
        self.size_threshold = size_threshold
        self.filter_params = filter_params
        self.consider_implicit = consider_implicit
        self._prepare_implicit()
        self.docs = pd.read_csv(docs_path)
        self.docs.fillna('', inplace=True)
        self.nr_docs = self.docs['filenr'].max()
        self.nr_passages = []
        self.passages_by_doc = []
        for doc_id in range(self.nr_docs):
            passages = self._doc_passages(doc_id+1)
            passages = self._filter_passages(passages)
            self.passages_by_doc.append(passages)
            self.nr_passages.append(len(passages))
        # Prepare caching of tuning hints
        self.doc_to_hints = {}
        # Calculate statistics
        self.asg_counts, self.param_counts = self._assignment_stats()
        # Sort hints by parameter
        self.param_to_hints = self._hints_by_param()
        # Output a summary of data read
        print(f'Initializing documents from file {docs_path} ...')
        print('Sample of tuning hints:')
        print(self.docs.sample())
        print(f'Nr. documents read: {self.nr_docs}')
        print(f'Nr. passages by doc: {self.nr_passages}')
        print(f'Nr. mentions per assignment: {self.asg_counts.most_common()}')
        print(f'Nr. documents per parameter: {self.param_counts.most_common()}')

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
            p_length += s_length
            if p_length > self.size_threshold:
                # Start new passage
                passages.append(" ".join(passage))
                passage = [snippet]
                p_length = 0
            else:
                # Append snippet to passage
                passage.append(snippet)
                p_length += s_length
        return passages
    
    def _enrich_passage(self, passage):
        """ Add implicit parameters and values to passage. """
        e = self.transformer.encode([passage], convert_to_tensor=True)[0]
        max_sim = -1
        max_param = None
        for p, p_e in zip(self.all_params, self.p_embeddings):
            sim = util.pytorch_cos_sim(e.unsqueeze(0), p_e.unsqueeze(0))[0][0]
            if sim > max_sim:
                max_sim = sim
                max_param = p
        if max_param:
            passage += f' {max_param} '
        passage += ' 1 0'
        return passage
    
    def _filter_passages(self, passages):
        """ Filter passages to potentially relevant ones. """
        # Filter based on simple string matching (need parameters and values)
        passages = [p for p in passages if get_parameters(p) and get_values(p)]
        # If available, use DBMS to filter to passages containing real parameters
        if self.dbms:
            passages = [p for p in passages if any(
                self.dbms.is_param(t) for t in get_parameters(p))]
        return passages
    
    def _prepare_implicit(self):
        """ Prepare extraction of implicit tuning hints. """
        if self.consider_implicit:
            self.transformer = SentenceTransformer('paraphrase-distilroberta-base-v1')
            self.all_params = self.dbms.all_params()
            self.p_embeddings = self.transformer.encode(
                self.all_params, convert_to_tensor=True)

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
                if self.consider_implicit:
                    exp_passage = self._enrich_passage(passage)
                    print(f'Enriched passage: {exp_passage}')
                else:
                    exp_passage = passage
                    
                params = re.finditer(r'[a-z_]+_[a-z]+', exp_passage)
                values = re.finditer(r'\d+[a-zA-Z]*%{0,1}', exp_passage)
                for param in params:
                    print(f'Param: {param}')
                    if not self.filter_params or self.dbms.is_param(param.group()):
                        for value in values:
                            hint = TuningHint(doc_id, passage, param, value)
                            hints.append(hint)
            self.doc_to_hints[doc_id] = hints
            return hints
        
    def _assignment_stats(self):
        """ Generate statistics on candidate parameter assignments. """
        asg_counter = Counter()
        param_counter = Counter()
        for doc_id in range(self.nr_docs):
            doc_asgs = set()
            doc_params = set()
            hints = self.get_hints(doc_id)
            for hint in hints:
                asg = (hint.param.group(), hint.value.group())
                doc_asgs.add(asg)
                doc_params.add(asg[0])
            for asg in doc_asgs:
                asg_counter.update([asg])
            for param in doc_params:
                param_counter.update([param])
        return asg_counter, param_counter
        
    def _hints_by_param(self):
        """ Maps parameters to corresponding hints. """
        param_to_hints = defaultdict(lambda: set())
        for doc_id, doc_hints in self.doc_to_hints.items():
            for hint in doc_hints:
                param = hint.param.group()
                param_to_hints[param].add((doc_id, hint))
        return param_to_hints