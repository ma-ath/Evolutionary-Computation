import numpy as np
from abc import ABC, abstractmethod

class Mutation(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def mutate(self, X):
        pass

class GaussianMutation(Mutation):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def mutate(self, X):
        noise = np.random.normal(loc=0, scale=self.sigma, size=X.shape)
        return X + noise

class BitWiseMutation(Mutation):
    def __init__(self, mutation_rate):
        super().__init__()
        self.mutation_rate = self.mutation_rate

    def mutate(self, X):
        mask = np.random.random_sample(X.shape) < self.mutation_rate
        return np.bitwise_xor(X, mask).astype(np.int32)
