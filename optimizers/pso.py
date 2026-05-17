import copy
import numpy as np
from base import PopulationOptimizer

class ParticleSwarmOptimizer(PopulationOptimizer):

    def __init__(self,
                 problem,
                 direction,
                 *,
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
                         population_size=population_size,
                         dimensions=dimensions,
                         bounds=bounds,
                         **kwargs)
        
        self.p_best = copy.deepcopy(self.X)
        self.p_best_fitness = copy.deepcopy(self.fitness)
        if velocity is None:
            self.V = np.zeros((self.population_size,self.dimensions))
        else:
            self.V = velocity
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def evolve(self):
        p_global_best, _ = self.get_best()
        r1 = np.random.uniform(low=0, high=1, size=(self.population_size, self.dimensions))
        r2 = np.random.uniform(low=0, high=1, size=(self.population_size, self.dimensions))
        self.V = self.w*self.V + self.c1*r1*(self.p_best - self.X) + self.c2*r2*(p_global_best - self.X)
        self.X = np.clip(self.X + self.V, a_min=self.bounds[:,0][:,np.newaxis], a_max=self.bounds[:,-1][:,np.newaxis])
        self.fitness = self.problem.evaluate(self.X)
        if self.direction == 'min':
            mask = self.fitness < self.p_best_fitness
        else:
            mask = self.fitness > self.p_best_fitness
        self.p_best_fitness = np.where(mask, self.fitness, self.p_best_fitness)
        self.p_best = np.where(mask[:,np.newaxis], self.X, self.p_best)
        return self.get_best()
        

        
        
