import numpy as np
from abc import ABC, abstractmethod

class SelectionOperator(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def select(self, population, fitness):
        pass

class TournamentSelection(SelectionOperator):
    def __init__(self, k=2):
        super().__init__()
        self.k = k

    def select(self, population, fitness):
        
        
