from abc import ABC, abstractmethod

class Processor(ABC):
    @abstractmethod
    def add_doc(self, doc, userId):
        pass

    @abstractmethod
    def save(self):
        pass
