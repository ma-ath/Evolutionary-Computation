import numpy as np
from abc import ABC, abstractmethod
from pymoo.problems import get_problem

class BenchmarkFunction(ABC):

    @abstractmethod
    def evaluate(self, X):
        raise NotImplementedError
    
class Sphere(BenchmarkFunction):

    def evaluate(self, X):
        return np.sum(X**2, axis=1)
    
class Rosenbrock(BenchmarkFunction):

    def evaluate(self, X):
        xi = X[:,:-1]
        xi_next = X[:,1:]
        return np.sum(100*(xi_next - xi**2)**2 + (1 - xi)**2, axis=1)

class Rastrigin(BenchmarkFunction):

    def evaluate(self, X, A=10):
        d = X.shape[1]
        return A*d + np.sum(X**2 - A*np.cos(2*np.pi*X), axis=1)
    
class Ackley(BenchmarkFunction):

    def evaluate(self, X, a=20, b=0.2, c=2*np.pi):
        d = X.shape[1]
        return -a*np.exp(-b*np.sqrt((1/d)*np.sum(X**2, axis=1))) - np.exp((1/d)*np.sum(np.cos(c*X), axis=1)) + a + np.exp(1)
    
class Himmelblau(BenchmarkFunction):

    def evaluate(self, X):
        assert X.shape[1] == 2, "Himmelblau function only takes two variables."
        return (X[:,0]**2 + X[:,1] - 11)**2 + (X[:,0] + X[:,1]**2 -7)**2

class PymooWrapper(BenchmarkFunction):
    def __init__(self, problem_name, n_var):
        self.problem = get_problem(problem_name, n_var=n_var)
        
    def evaluate(self, X):
        return self.problem.evaluate(X)