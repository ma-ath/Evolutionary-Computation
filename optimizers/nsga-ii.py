import numpy as np
from base import MultiObjectiveOptimizer

class NSGAII (MultiObjectiveOptimizer):

    def __init__(self,
                 problem,
                 *,
                 population_size,
                 dimensions,
                 bounds,
                 **kwargs):
        super().__init__(problem,
                         population_size=population_size,
                         dimensions=dimensions,
                         bounds=bounds,
                         **kwargs)
        
    def evolve(self):
        pass

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
                
            # 3. Calcular a distância para os indivíduos intermediários usando fatiamento (slicing)
            # A fórmula é: dist_{i} += (fitness_{i+1} - fitness_{i-1}) / (f_max - f_min)
            prev_idx = sorted_idx[:-2]   # Índices i-1
            next_idx = sorted_idx[2:]    # Índices i+1
            current_idx = sorted_idx[1:-1] # Índices i
            
            dist_m = (front_fitness[next_idx, m] - front_fitness[prev_idx, m]) / (f_max - f_min)
            distances[current_idx] += dist_m
            
        return distances
        
    