from tqdm.auto import tqdm
from benchmark import *
from optimizers.pso import ParticleSwarmOptimizer

problem = Ackley()
optimizer = ParticleSwarmOptimizer(problem=problem,
                                   direction='min',
                                   population_size=100,
                                   dimensions=2,
                                   bounds=[-5,5])

fitness = []
for generation in tqdm(range(1,150+1)):
    best_pos, best_fit = optimizer.evolve()
    fitness.append(best_fit)
    print(f"Generation: {generation} | Best fitness: {best_fit}")
