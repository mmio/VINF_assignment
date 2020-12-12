import json
from collections import Counter
from processors.processor import Processor
import matplotlib.pyplot as plt

class HistogramOfTokens(Processor):
    def __init__(self, destination):
        self.counter = Counter()
        self.destination = destination
    
    def add_doc(self, doc):
        self.counter.update(doc.split())

    def save(self):
        with open(self.destination, 'w') as fp:
            json.dump(self.counter, fp)

        plt.figure(figsize=(30,8))
        plt.bar(*zip(*self.counter.most_common(150)), width=.4)
        plt.xticks(rotation=90)
        plt.savefig('histogramOfTokens.pdf', bbox_inches='tight')
        plt.close() #, is this a thing?

    def get_items(self):
        return self.counter.items()