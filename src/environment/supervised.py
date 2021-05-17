'''
Created on Apr 18, 2021

@author: immanueltrummer
'''
from doc.collection import DocCollection
from environment.common import DocTuning
from environment.common import DecisionType
import pandas as pd
from random import shuffle
from environment.bert_tuning import TuningBertFine

#class LabeledDocTuning(DocTuning):
class LabeledDocTuning(TuningBertFine):
    """ Supervised training for extracting tuning hints. """

    def __init__(self, docs: DocCollection, nr_hints, label_path):
        """ Initialize from given tuning documents, database, and benchmark. 
        
        Args:
            docs: collection of text documents with labeled hints
            label_path: path to file with labels for tuning hints
            nr_hints: consider so many hints during optimization
        """
        super().__init__(docs)
        self.hints = self._hints_by_param()
        shuffle(self.hints)
        self.nr_hints = min(len(self.hints), nr_hints)
        self.ok_actions = self._read_labels(label_path)
        self.reset()
        
    def _hints_by_param(self):
        """ Prioritize hints in tuning document collection. """
        ordered_hints = []
        for param, _ in self.docs.param_counts.most_common():
            param_hints = self.docs.param_to_hints[param]
            ordered_hints += param_hints
        return ordered_hints
    
    def _read_labels(self, label_path):
        """ Read labels associated with tuning hints. """
        ok_actions = {}
        df = pd.read_csv(label_path)
        print(df.sample(10))
        for i in range(self.nr_hints):
            _, hint = self.hints[i]
            passage = hint.passage
            param = hint.param.group()
            value = hint.value.group()
            labels = df.loc[(df['sentence']==passage) & \
                          (df['parameter']==param) & \
                          (df['value']==value)]
            nr_labels = labels.shape[0]
            if nr_labels == 1:
                ok_actions[(i, 0)] = [int(labels['base'].iloc[0])]
                ok_actions[(i, 1)] = [int(a) for a in labels['operators'].iloc[0].split(';')]
        return ok_actions

    def _take_action(self, action):
        """ Process action and return obtained reward. """
        # Generate list of desirable actions
        index = (self.hint_ctr, self.decision)
        desirable = []
        if index in self.ok_actions:
            desirable = self.ok_actions[index]
        elif self.decision == DecisionType.PICK_BASE:
            desirable = [4]
        # Reward taken action accordingly
        if action in desirable:
            return 1
        else:
            return -1
    
    def _finalize_episode(self):
        """ Nothing to do. """
        return 0

    def _reset(self):
        """ Initializes for new tuning episode. """
        self.label = None