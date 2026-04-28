import numpy as np
from base import PopulationOptimizer


class GeneticAlgorithm(PopulationOptimizer):
    def __init__(self, problem, direction, *, population_size, dimensions, bounds, selection, crossover, mutation, survival_strategy, **kwargs):
        super().__init__(problem, direction, population_size=population_size, dimensions=dimensions, bounds=bounds, **kwargs)
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.survival_strategy = survival_strategy

    def evolve(self):
        population_fitness = self.problem.evaluate(self.X)
        selected_population = self.selection.select(self.X, population_fitness)
        offspring = self.mutation(self.crossover(selected_population))
        self.X = self.survival_strategy(self.X, offspring)

class BinaryGeneticAlgorithm(GeneticAlgorithm):
    def __init__(self,
                 problem,
                 direction,
                 *,
                 population_size,
                 dimensions,
                 bounds,
                 selection,
                 crossover,
                 mutation,
                 survival_strategy,
                 **kwargs):
        super().__init__(problem,
                         direction,
                         population_size=population_size,
                         dimensions=dimensions,
                         bounds=bounds,
                         selection=selection, 
                         crossover=crossover,
                         mutation=mutation,
                         survival_strategy=survival_strategy,
                         **kwargs)
        self.X = np.random.randint(low=0, high=2, size=self.X.size)

        

        
