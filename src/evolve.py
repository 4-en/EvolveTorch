# main file

import numpy as np
import base64
import random
import torch
import tqdm

import matplotlib.pyplot as plt

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
    def __init__(self, model: torch.nn.Module, parents = [], dna = None, population = None):
        self.model = model
        self.fitness = 0.0
        if dna is not None:
            self.dna = dna
        else:
            self.dna = DNA()
        self.parents = parents
        self.population = population
    
    def __str__(self) -> str:
        return f"Genome of {self.model.__class__.__name__} with DNA {self.dna}"
    
    def mutate(self, mutation_rate=0.20, mutation_amount=2)->"Genome":
        """Returns a new instance of this genome with a mutated DNA and model"""
        # by default, the mutation should be between half and double the current value, or *-1 of it
        # [-mutation_amount, -1/mutation_amount] U [1/mutation_amount, mutation_amount
        # currently it is between -mutation_amount and +mutation_amount
        # TODO: avoid multiplying by values too close to 0
        parents = [self.dna]
        dna = self.dna.mutate()
        model = self.population.factory()
        # mutate the model
        for param_old, param_new in zip(self.model.parameters(), model.parameters()):
            chances = torch.rand(param_old.shape)
            mask = chances < mutation_rate
            mutations = (torch.randn(param_old.shape)*2-1) * mask * mutation_amount
            param_new.data = param_old * mutations
        return Genome(model, parents, dna, population=self.population)
    
    def crossover(self, other:"Genome")->"Genome":
        """Returns a new instance of this genome with a crossover of the DNA and model"""
        parents = [self.dna, other.dna]
        dna = DNA()
        model = self.population.factory()
        # crossover the model
        for param_self, param_other, param_new in zip(self.model.parameters(), other.model.parameters(), model.parameters()):
            chances = torch.rand(param_self.shape)
            mask = chances < 0.5
            param_new.data = param_self * mask + param_other * (~mask)
        return Genome(model, parents, dna, population=self.population)
    
    


class Population:
    """
    Represents a population of genomes.
    
    During training, each genome is evaluated and assigned a fitness score.
    The genomes are then sorted by fitness and the top genomes are selected to create the next generation.
    The genomes are then crossed over and mutated to create the next generation.
    """

    def __init__(self, factory: callable, fitness_function: callable, inverse_fitness=False, size: int = 128, weights: dict = None):
        """
        Create a population of genomes
        
        Parameters
        ----------
        factory : callable
            a function that returns a new instance of a genome
        fitness_function : callable
            a function that returns the fitness of a genome
        inversed_fitness : bool, optional
            true if the fitness function returns a value that is better the lower it is, by default False
        size : int, optional
            the size of the population, by default 128
        weights : dict, optional
            the weights of the different operations to create a new generation, by default None
        """
        self.factory = factory
        self.fitness_function = fitness_function
        self.inverse_fitness = inverse_fitness
        self.size = size
        self.generation = 0
        self.keep_old_gen_n = 0 # number of old models to keep
        self.top_p = 0.2 # percentage of top genomes to use for the next generation
        self.top_weighting = lambda idx: 1 / 1 + idx # weighting of the top genomes
        self.generations = [] # old generations, usually without the models
        self.genomes = []
        self.verbose = True # print progress
        self.weights = {
            "mutation": 20,
            "crossover": 10,
            "elitism": 5,
            "random": 10
        }
        if weights is not None:
            self.weights.update(weights)

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
    
    def set_weights(self, weights: dict):
        """
        Set the weights of the different operations to create a new generation
        
        keys:
        - mutation
        - crossover
        - elitism
        - random
        """
        self.weights.update(weights)

    def _select_operation(self):
        """
        Returns an operation to create a new generation based on the weights"""
        
        keys = list(self.weights.keys())
        weights = np.array(list(self.weights.values()))
        weights = weights / weights.sum()
        return np.random.choice(keys, p=weights)
    
    
    def next_generation(self):
        "create the next generation of genomes based on the current population"
        self.generation += 1
        old_genomes = self.genomes
        self.genomes = []

        # sort the genomes by fitness
        #old_genomes.sort(key=lambda g: g.fitness, reverse=True)

        next_elitism = 0

        top_pop = int(self.top_p * self.size)
        top_pop = old_genomes[:top_pop]
        top_weights = [self.top_weighting(i) for i in range(len(top_pop))]

        for _ in range(self.size):
            # select an operation
            operation = self._select_operation()

            new_genome = None


            if operation == "random":
                # create a random genome
                new_genome = self.create_genome()
            elif operation == "mutation":
                # mutate a random genome
                mutation_target = random.choices(top_pop, weights=top_weights)[0]
                new_genome = mutation_target.mutate()
            elif operation == "crossover":
                # crossover two random genomes
                parent1 = random.choices(top_pop, weights=top_weights)[0]
                weight_no_parent1 = [w * int(parent1==top_pop[i]) for i, w in enumerate(top_weights)] # set the weight of parent1 to 0
                parent2 = random.choices(top_pop, weights=weight_no_parent1)[0]
                new_genome = parent1.crossover(parent2)
            elif operation == "elitism":
                # use the next genome in the top genomes
                if next_elitism < len(top_pop):
                    new_genome = top_pop[next_elitism]
                    next_elitism += 1
                else:
                    new_genome = self.create_genome()
            else:
                raise ValueError(f"Unknown operation {operation}")
            
                
            self.genomes.append(new_genome)

        save_top = self.keep_old_gen_n
        for genome in old_genomes:
            if save_top > 0:
                save_top -= 1
                continue
            #genome.model = None
            # maybe use del ?

        self.generations.append(old_genomes)


    def fill_population(self):
        missing = self.size - len(self.genomes)
        new_genomes = [self.create_genome() for _ in range(missing)]
        self.genomes.extend(new_genomes)
    
    def create_genome(self):
        g = Genome(self.factory(), population=self)
        g.model.requires_grad_(False)
        return g
            
    
    def _print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def eval_fitness(self):
        with torch.no_grad():
            for genome in self.genomes:
                genome.fitness = self.fitness_function(genome)
    
    def evolve(self, generations=1):

        if self.generation == 0:
            self._print(f"Evaluating generation {self.generation}")
            # evaluate the fitness of the genomes
            for genome in tqdm.tqdm(self.genomes):
                genome.fitness = self.fitness_function(genome)

        for _ in range(generations):

            self._print(f"Creating generation {self.generation+1}")
            # create the next generation
            self.next_generation()

            self._print(f"Evaluating generation {self.generation}")
            # evaluate the fitness of the genomes
            for genome in tqdm.tqdm(self.genomes):
                genome.fitness = self.fitness_function(genome)

            # sort the genomes by fitness
            self.genomes.sort(key=lambda g: g.fitness, reverse=(not self.inverse_fitness))
            # print best 5 fitness
            self._print(f"Best fitness: {[g.fitness for g in self.genomes[:5]]}")

    def plot_fitness(self):
        plt.plot([g.fitness for g in self.genomes])
        plt.show()

    def get_best_model(self):
        return self.genomes[0].model


        
class TestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 32)
        self.ReLU = torch.nn.ReLU()
        #self.linear2 = torch.nn.Linear(32, 32)
        self.linear3 = torch.nn.Linear(32, 1)
    
    def forward(self, x):
        x = self.linear(x)
        x = self.ReLU(x)
        #x = self.linear2(x)
        #x = self.ReLU(x)
        x = self.linear3(x)
        return x

def fitness_func(genome: Genome):
    x = torch.arange(0, 2*np.pi, 0.1)
    x = x.reshape(-1, 1)
    y = torch.cos(x)
    y_hat = genome.model(x)
    loss = torch.nn.functional.mse_loss(y_hat, y)
    return 1.0/loss

def tests():
    print("Running tests")
    print("Creating a population of 128 genomes")
    p = Population(lambda: TestModel(), fitness_func, size=128)
    print(p)
    print("Creating a genome")
    g = Genome(TestModel())
    print(g)
    print("Done")

    p.evolve(50)

if __name__ == "__main__":
    tests()