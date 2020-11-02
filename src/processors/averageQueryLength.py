import json
from collections import Counter
from processors.processor import Processor
import matplotlib.pyplot as plt

class AverageQueryLength(Processor):
    def __init__(self, destination):
        self.numberOfQueries = 0
        self.queryLengths = 0
        self.destination = destination
    
    def add_doc(self, text, userId):
        self.numberOfQueries += 1
        self.queryLengths += len(text)

    def save(self):
        with open(f'{self.destination}averageQueryLength.txt', 'w') as fp:
            fp.write(str(self.queryLengths / self.numberOfQueries))