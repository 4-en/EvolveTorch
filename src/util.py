# this files contains some useful optional functions and classes
from src.evolve import Genome
import copy

class CloneModelFactory:
    """
    clones the model and returns it to fill the population
    """
    def __init__(self, model):
        self.model = model
    
    def __call__(self):
        return copy.deepcopy(self.model)
    
class MutateModelFactory:
    """
    mutates the model and returns it to fill the population
    """
    def __init__(self, model, mutation_rate=0.1, mutation_strength=2):
        self.genome = Genome(model)
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
    
    def __call__(self):
        mutant = self.genome.mutate(self.mutation_rate, self.mutation_strength)
        return mutant.model
    
