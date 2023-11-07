# main file

import numpy as np
import base64
import random
import torch

class DNA:
    """
    Represents a genome as a sequence of bytes.
    Can be mutated and crossed over.
    """
    DNA_BYTES = 8
    def __init__(self, data=None):
        if data is None:
            self.dna = self.create_dna()
        else:
            self.dna = data
    
    def create_dna(self)->np.number:
        # create random bytes
        b = bytearray(random.getrandbits(8) for _ in range(self.DNA_BYTES))
        return b
    
    def __str__(self) -> str:
        b64s = base64.b64encode(self.dna).decode('utf-8')
        return b64s
    
    def __repr__(self) -> str:
        return f"DNA({self.dna})"
    
    def mutate(self)->"DNA":
        "return a mutated copy of this DNA"
        index = random.randint(0, len(self.dna)*8-1)
        # copy the dna
        new_dna = bytearray(self.dna)
        # mutate the dna
        # flip a random bit
        new_dna[index//8] ^= 1 << (index%8)
        # return a new DNA
        return DNA(new_dna)

        

class Genome:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.fitness = 0.0
        
        self.dna = DNA()
    
    def __str__(self) -> str:
        return f"Genome of {self.model.__class__.__name__} with DNA {self.dna}"
    
    


class Population:
    def __init__(self, factory: callable, size: int = 128):
        self.factory = factory
        self.size = size
        self.generation = 0
        self.genomes = []

        self.fill_population()

    def __len__(self):
        return len(self.genomes)
    
    def __getitem__(self, index):
        return self.genomes[index]
    
    def __setitem__(self, index, value):
        self.genomes[index] = value

    def __iter__(self):
        return iter(self.genomes)
    
    def __next__(self):
        return next(self.genomes)
    
    def __str__(self):
        return f"Population of {self.size} genomes"
    
    def __repr__(self):
        return f"Population(factory={self.factory}, size={self.size})"
    
    def fill_population(self):
        missing = self.size - len(self.genomes)
        new_genomes = [self.create_genome() for _ in range(missing)]
        self.genomes.extend(new_genomes)
    
    def create_genome(self):
        return Genome(self.factory())

    def evolve(self):
        pass

    def crossover(self):
        pass

    def mutate(self):
        pass

    def evaluate(self):
        pass

    def select(self):
        pass

    def get_best(self):
        pass


def tests():
    print("Running tests")
    print("Creating a population of 128 genomes")
    p = Population(lambda: torch.nn.Linear(1, 1))
    print(p)
    print("Creating a genome")
    g = Genome(torch.nn.Linear(1, 1))
    print(g)
    print("Done")

if __name__ == "__main__":
    tests()