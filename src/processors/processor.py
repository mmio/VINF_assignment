from abc import ABC, abstractmethod

class Processor(ABC):
    @abstractmethod
    def add_doc(self, doc):
        pass

    @abstractmethod
    def save(self):
        pass
