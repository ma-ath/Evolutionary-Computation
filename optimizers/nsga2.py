import numpy as np
from optimizers.base import MultiObjectiveOptimizer
from components.selections import SelectionOperator
from components.crossover import CrossOverOperator
from components.mutations import Mutation

class NSGAII (MultiObjectiveOptimizer):

    def __init__(self,
                 problem,
                 *,
                 population_size,
                 dimensions,
                 bounds,
                 crossover_operator: CrossOverOperator,
                 selection_operator: SelectionOperator,
                 mutation_operator: Mutation | None,
                 **kwargs):
        super().__init__(problem,
                         population_size=population_size,
                         dimensions=dimensions,
                         bounds=bounds,
                         **kwargs)

        self.crossover_operator = crossover_operator
        self.selection_operator = selection_operator
        self.mutation_operator = mutation_operator
        fronts = self.fast_non_dominated_sort(self.fitness)
        self.ranks = np.zeros(self.population_size)
        self.distances = np.zeros(self.population_size)

        current_rank = 1
        for front in fronts:
            self.ranks[front] = current_rank
            self.distances[front] = self.calculate_crowding_distance(self.fitness, front)
            current_rank += 1

    def calculate_crowding_distance(self, fitness, front):

        l = len(front)
        distances = np.zeros(l)

        if l == 0:
            return distances
        elif l <= 2:
            return np.full(l, np.inf)

        front_fitness = fitness[front]
        M = front_fitness.shape[1]
 
        for m in range(M):

            sorted_idx = np.argsort(front_fitness[:, m])
            
            distances[sorted_idx[0]] = np.inf
            distances[sorted_idx[-1]] = np.inf
            
            f_min = front_fitness[sorted_idx[0], m]
            f_max = front_fitness[sorted_idx[-1], m]

            # Avoiding zero division in the case that all the solutions share the same value
            # on this objective function
            if f_max == f_min:
                continue
                
            # dist_{i} += (fitness_{i+1} - fitness_{i-1}) / (f_max - f_min)
            prev_idx = sorted_idx[:-2] 
            next_idx = sorted_idx[2:]   
            current_idx = sorted_idx[1:-1] 
            
            dist_m = (front_fitness[next_idx, m] - front_fitness[prev_idx, m]) / (f_max - f_min)
            distances[current_idx] += dist_m
            
        return distances
    
    def environmental_selection(self, combined_X, combined_fitness, N):
        """
            combined_X: (2N,D) matrix of parents + offspring
            combined_fitness: (2N, M) matrix representing the fitness of parents + offspring
            N: Size of the population for the next generation
        """

        fronts = self.fast_non_dominated_sort(combined_fitness)
        
        next_X = np.zeros((N, combined_X.shape[1]))
        next_fitness = np.zeros((N, combined_fitness.shape[1]))
        next_ranks = np.zeros(N)
        next_distances = np.zeros(N)
        
        current_size = 0
        current_rank = 1
        
        for front in fronts:

            front_size = len(front)
            distances = self.calculate_crowding_distance(combined_fitness, front)
            
            if current_size + front_size <= N:
                next_X[current_size : current_size + front_size] = combined_X[front]
                next_fitness[current_size : current_size + front_size] = combined_fitness[front]
                next_ranks[current_size : current_size + front_size] = current_rank
                next_distances[current_size : current_size + front_size] = distances
                
                current_size += front_size
                current_rank += 1
                
            else:
                remaining_space = N - current_size

                sorted_local_idx = np.argsort(distances)[::-1]
                best_local_idx = sorted_local_idx[:remaining_space]
                
                best_global_idx = np.array(front)[best_local_idx]
                best_distances = distances[best_local_idx]

                next_X[current_size:] = combined_X[best_global_idx]
                next_fitness[current_size:] = combined_fitness[best_global_idx]
                next_ranks[current_size:] = current_rank
                next_distances[current_size:] = best_distances
                
                break
                
        return next_X, next_fitness, next_ranks, next_distances

    def evolve(self):
 
        mating_pool = self.selection_operator.select(self.X, self.ranks, self.distances)

        parents_2 = np.random.permutation(mating_pool)
        offspring = self.crossover_operator.crossover(mating_pool, parents_2)
        if self.mutation_operator is not None:
            offspring = self.mutation_operator.mutate(offspring)

        offspring = np.clip(offspring, self.bounds[:, 0], self.bounds[:, 1])
        fitness_offspring = self.problem.evaluate(offspring)

        combined_X = np.vstack((self.X, offspring))
        combined_fitness = np.vstack((self.fitness, fitness_offspring))
        
        self.X, self.fitness, self.ranks, self.distances = self.environmental_selection(
            combined_X, combined_fitness, self.population_size
        )
        
    