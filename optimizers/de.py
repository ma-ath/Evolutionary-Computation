import numpy as np
from .base import PopulationOptimizer
from components.selections import SelectionOperator
from components.crossover import CrossOverOperator

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
                 crossover_operator: CrossOverOperator,
                 selection_operator: SelectionOperator,
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
        assert self.differences in [1,2], "The number of differences to be used must be either 1 or 2."
        self.crossover_operator = crossover_operator
        self.selection_operator = selection_operator
    
    def evolve(self):
        
        r1_idx = np.random.randint(low=0, high=self.population_size, size=self.population_size)
        r2_idx = np.random.randint(low=0, high=self.population_size, size=self.population_size)
        r3_idx = np.random.randint(low=0, high=self.population_size, size=self.population_size)
        i_idx = np.arange(self.population_size)

        while np.any(r1_idx == i_idx):
            mask = (r1_idx == i_idx)
            r1_idx[mask] = np.random.randint(low=0, high=self.population_size, size=np.sum(mask))

        while np.any((r2_idx == i_idx) | (r2_idx == r1_idx)):
            mask = ((r2_idx == i_idx) | (r2_idx == r1_idx))
            r2_idx[mask] = np.random.randint(low=0, high=self.population_size, size=np.sum(mask)) 

        while np.any((r3_idx == i_idx) | (r3_idx == r1_idx) | (r3_idx == r2_idx)):
            mask = ((r3_idx == i_idx) | (r3_idx == r1_idx) | (r3_idx == r2_idx))
            r3_idx[mask] = np.random.randint(low=0, high=self.population_size, size=np.sum(mask))

        if self.differences > 1:
            r4_idx = np.random.randint(low=0, high=self.population_size, size=self.population_size)
            r5_idx = np.random.randint(low=0, high=self.population_size, size=self.population_size)

            while np.any((r4_idx == i_idx) | (r4_idx == r1_idx) | (r4_idx == r2_idx) | (r4_idx == r3_idx)):
                mask = ((r4_idx == i_idx) | (r4_idx == r1_idx) | (r4_idx == r2_idx) | (r4_idx == r3_idx))
                r4_idx[mask] = np.random.randint(low=0, high=self.population_size, size=np.sum(mask))
            
            while np.any((r5_idx == i_idx) | (r5_idx == r1_idx) | (r5_idx == r2_idx) | (r5_idx == r3_idx) | (r5_idx == r4_idx)):
                mask = ((r5_idx == i_idx) | (r5_idx == r1_idx) | (r5_idx == r2_idx) | (r5_idx == r3_idx) | (r5_idx == r4_idx))
                r5_idx[mask] = np.random.randint(low=0, high=self.population_size, size=np.sum(mask))
            mutant_v = self.X[r1_idx] + self.F*((self.X[r2_idx] - self.X[r3_idx]) + (self.X[r4_idx] - self.X[r5_idx]))

        else:
            mutant_v = self.X[r1_idx] + self.F*(self.X[r2_idx] - self.X[r3_idx])

        trial_v = np.clip(self.crossover_operator.crossover(self.X, mutant_v), a_min=self.bounds[:,0], a_max=self.bounds[:,-1])
        fitness = self.problem.evaluate(trial_v)
        self.X, self.fitness = self.selection_operator.select(direction=self.direction,
                                                              X=self.X,
                                                              fitness_X=self.fitness,
                                                              Y=trial_v,
                                                              fitness_Y=fitness)
        return self.get_best()