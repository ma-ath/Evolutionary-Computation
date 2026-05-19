import numpy as np
from abc import ABC, abstractmethod

class SelectionOperator(ABC):

    @abstractmethod
    def select(self, *, direction, X, fitness_X, Y, fitness_Y):
        pass

class GreedySelection(SelectionOperator):
    
    def select(self, *, direction, X, fitness_X, Y, fitness_Y):
        if direction == 'min':
            mask = fitness_Y < fitness_X
        else:
            mask = fitness_Y > fitness_X
        return np.where(mask[:,np.newaxis], Y, X), np.where(mask, fitness_Y, fitness_X)

class TournamentSelection(SelectionOperator):
    def __init__(self, k=2):
        super().__init__()
        self.k = k

    def select(self, population, fitness):
        population_fitness = np.concatenate((population,fitness),axis=1)
        counter = population.shape[0]/self.k
        mating_pool = []
        while counter != 0:
            tournament_groups = self.rng.choice(population_fitness, size=(population.shape[0]/self.k, self.k), replace=False, shuffle=True)
            mating_pool.append(tournament_groups)
            counter-=counter
        mating_pool = np.array(mating_pool)
        selected_mating_pool = []
        for group in mating_pool:
            best_fitness_set = []
            best_fitness_idx = 0
            for i in range(self.k):
                if group[i][2] > group[best_fitness_idx][2]:
                    best_fitness_idx = i
                elif group[i][2] == group[best_fitness_idx][2] and i != 0:
                    best_fitness_set.append(best_fitness_idx)
                    best_fitness_idx = i
                    if i == self.k-1:
                        best_fitness_set.append(best_fitness_idx)
            selected_mating_pool.append(self.rng.choice(best_fitness_set, size=1, replace=False, shuffle=True))
        selected_mating_pool = np.array(selected_mating_pool)
                
        
