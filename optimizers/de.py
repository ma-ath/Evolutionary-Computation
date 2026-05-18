import numpy as np
from base import PopulationOptimizer

class DifferentialEvolution(PopulationOptimizer):
    def __init__(self,
                 problem,
                 direction,
                 *,
                 population_size,
                 dimensions,
                 bounds,
                 F = 0.5,
                 mutation,
                 differences = 1,
                 crossover,
                 **kwargs):
        super().__init__(problem,
                         direction,
                         population_size=population_size,
                         dimensions=dimensions,
                         bounds=bounds,
                         **kwargs)
        
        self.F = F
        self.mutation = mutation
        self.differences = differences
        self.crossover = crossover
    
    def evolve(self):
        
        r1_idx = np.random.randint(low=0, high=self.population_size, size=self.population_size)
        r2_idx = np.random.randint(low=0, high=self.population_size, size=self.population_size)
        r3_idx = np.random.randint(low=0, high=self.population_size, size=self.population_size)
        i_idx = np.arange(self.population_size)

        while np.any(r1_idx == i_idx):
            mask = r1_idx == i_idx
            r1_idx[mask] = np.random.randint(low=0, high=self.population_size, size=np.sum(mask))

        while np.any(r2_idx == i_idx | r2_idx == r1_idx):
            mask = (r2_idx == i_idx | r2_idx == r1_idx):
            r2_idx[mask] = np.random.randint(low=0, high=self.population_size, size=np.sum(mask))  