import json
from collections import Counter
from processors.processor import Processor
import matplotlib.pyplot as plt
from collections import defaultdict

class AverageNumberOfQueriesPerUser(Processor):
    def __init__(self, destination):
        self.numberOfQueries = 0
        self.seenUsers = defaultdict(bool)
        self.destination = destination
    
    def add_doc(self, text, userId):
        self.numberOfQueries += 1
        
        if not self.seenUsers.get(userId, False):
            self.seenUsers.update({userId: True})

    def save(self):
        with open(f'{self.destination}averageNumberOfQueriesPerUser.txt', 'w') as fp:
            fp.write(str(self.numberOfQueries / len(self.seenUsers.values())))