import numpy as np
from abc import ABC, abstractmethod

class CrossOverOperator(ABC):
    def __init__(self,
                 probability):
        super().__init__()

        self.probability = probability

    @abstractmethod
    def crossover(self, X, Y):
        pass

class BinomialCrossOver(CrossOverOperator):
    def __init__(self,
                 probability):
        super().__init__(probability)

    def crossover(self, X, V):
        mask = np.random.uniform(low=0, high=1, size=(X.shape)) < self.probability
        j_rand = np.random.randint(low=0, high=X.shape[1], size=X.shape[0])
        mask[np.arange(X.shape[0]), j_rand] = True
        return np.where(mask, V, X)

