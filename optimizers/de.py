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
                 **kwargs):
        super().__init__(problem,
                         direction,
                         population_size=population_size,
                         dimensions=dimensions,
                         bounds=bounds,
                         **kwargs)
    
    def evolve(self):
        pass