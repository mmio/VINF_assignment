import json
from collections import Counter
from processors.processor import Processor

class HistogramOfTokens(Processor):
    def __init__(self, destination):
        self.counter = Counter()
        self.destination = destination
    
    def add_doc(self, doc, userId):
        self.counter.update([token.text for token in doc])

    def save(self):
        with open(self.destination, 'w') as fp:
            json.dump(self.counter, fp)

    def get_items(self):
        return self.counter.items()