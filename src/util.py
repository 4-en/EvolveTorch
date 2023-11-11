# this files contains some useful optional functions and classes
from evolve import Genome

class CloneModelFactory:
    """
    clones the model and returns it to fill the population
    """
    def __init__(self, model):
        self.model = model
    
    def __call__(self):
        return self.model.clone()
    
class MutateModelFactory:
    """
    mutates the model and returns it to fill the population
    """
    def __init__(self, model, mutation_rate=0.01, mutation_strength=2):
        self.genome = Genome(model)
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
    
    def __call__(self):
        mutant = self.genome.mutate(self.mutation_rate, self.mutation_strength)
    
