import copy
import numpy as np
from base import PopulationOptimizer

class ParticleSwarmOptimization(PopulationOptimizer):

    def __init__(self,
                 problem,
                 direction,
                 *,
                 generations,
                 population_size,
                 dimensions,
                 bounds,
                 velocity = None,
                 w=0.75,
                 c1=0.5,
                 c2=0.5,
                 **kwargs):
        super().__init__(problem,
                         direction,
                         generations=generations,
                         population_size=population_size,
                         dimensions=dimensions,
                         bounds=bounds,
                         **kwargs)
        
        self.p_best = copy.deepcopy(self.X)
        if velocity is None:
            self.V = np.zeros(population_size)
        else:
            self.V = velocity
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def evolve(self):
        
        
