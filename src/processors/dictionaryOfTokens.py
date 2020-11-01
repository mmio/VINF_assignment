import pickle
from processors.processor import Processor
from collections import defaultdict

class DictionaryOfTokens(Processor):
    def __init__(self, destination):
        self.dictionary = defaultdict(set)
        self.destination = destination
    
    def add_doc(self, doc, userId):
        for token in doc:
            ## keep keywords only, needs ner or tagger?
            ## save queries, not only users
            if token.text == '\n' or token.is_stop:
                continue

            try:
                self.dictionary[token.text].add(userId)
            except KeyError:
                self.dictionary[token.text] = {userId}

    def save(self):
        with open(self.destination, 'wb') as handle:
            pickle.dump(self.dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self):
        with open(self.destination, 'rb') as handle:
            self.dictionary = pickle.load(handle)

    def get_items(self):
        return self.dictionary.items()

    def get_item(self, key):
        return self.dictionary[key]