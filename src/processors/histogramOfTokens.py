import json
from collections import Counter
from processors.processor import Processor
import matplotlib.pyplot as plt

class HistogramOfTokens(Processor):
    def __init__(self, destination):
        self.counter = Counter()
        self.counter2 = Counter()
        self.destination = destination
    
    def add_doc(self, doc, userId=0):
        self.counter.update([token.text for token in doc])
        self.counter2.update([token.lemma_ for token in doc])

    def save(self):
        with open(self.destination, 'w') as fp:
            json.dump(self.counter, fp)

        with open(f'{self.destination}.normalized', 'w') as fp:
            json.dump(self.counter2, fp)

        plt.figure(figsize=(30,8))
        plt.bar(*zip(*self.counter.most_common(150)), width=.4)
        plt.xticks(rotation=90)
        plt.savefig('histogramOfTokens.pdf', bbox_inches='tight')

        plt.figure(figsize=(30,8))
        plt.bar(*zip(*self.counter2.most_common(150)), width=.4)
        plt.xticks(rotation=90)
        plt.savefig('histogramOfNormalizedTokens.pdf', bbox_inches='tight')

    def get_items(self):
        return self.counter.items()