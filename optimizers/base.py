import numpy as np
from abc import ABC, abstractmethod
from warnings import warn

class PopulationOptimizer(ABC):

    def __init__(self, problem, direction, *, generations, population_size, dimensions, bounds, **kwargs):
        super().__init__()
        self.problem = problem
        self.direction = direction.lower()
        if self.direction not in ['max', 'min']:
            warn("Direction not specified. Must be either 'min' or 'max'. Defaulting to 'min'")
            self.direction = 'min'
        self.generations = generations
        self.population_size = population_size
        self.dimensions = dimensions
        self.bounds = np.array(bounds)
        self.X = np.random.uniform(low=self.bounds[:,0], high=self.bounds[:,1], size=(population_size, dimensions))
        self.fitness = self.problem.evaluate(self.X)

    @abstractmethod
    def evolve(self):
        pass

    def get_best(self):
        if self.direction == 'max':
            idx = np.argmax(self.fitness)
        else:
            idx = np.argmin(self.fitness)
        return self.X[idx], self.fitness[idx]