# main file

import numpy as np
import random
import torch
import tqdm
import copy

import matplotlib.pyplot as plt


class NoWith:
    def __enter__(self):
        pass
    def __exit__(self, *args):
        pass

class DNA:
    """
    Represents a genome as a sequence of bytes.
    Can be mutated and crossed over.
    """
    DNA_LENGHT = 32
    DNA_CHARS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+/="
    def __init__(self, data=None):
        if data is None:
            self.dna = self.create_dna()
        else:
            self.dna = data
    
    def create_dna(self)->str:
        # create random string
        dna = "".join(random.choices(self.DNA_CHARS, k=self.DNA_LENGHT))
        return dna

    
    def __str__(self) -> str:
        return self.dna
    
    def __repr__(self) -> str:
        return f"DNA({self.dna})"
    
    def mutate(self, mutations=1)->"DNA":
        "return a mutated copy of this DNA"
        # replace a random character with a random character
        dna = self.dna
        for _ in range(mutations):
            idx = random.randint(0, len(dna)-1)
            dna = dna[:idx] + random.choice(self.DNA_CHARS) + dna[idx+1:]
        return DNA(dna)
    
    def crossover(self, other:"DNA")->"DNA":
        "return a crossover of this DNA and another DNA"
        # copy the dna
        new_dna = []
        # crossover the dna by choosing randomly from each dna
        for i in range(len(self.dna)):
            new_dna.append(random.choice([self.dna[i], other.dna[i]]))

        new_dna = "".join(new_dna)

        # return a new DNA
        return DNA(new_dna)
    
    def similarity(self, other:"DNA")->float:
        "return the similarity between this DNA and another DNA"
        # count the number of characters that are the same
        count = 0
        for b1, b2 in zip(self.dna, other.dna):
            if b1 == b2:
                count += 1
        return count / len(self.dna)

        

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

    def get_model(self):
        return self.model
    
    def __str__(self) -> str:
        return f"Genome of {self.model.__class__.__name__} with DNA {self.dna}"
    
    def mutate(self, mutation_rate=0.02, mutation_amount=2, epsilon=0.01, weight_decay=False)->"Genome":
        """Returns a new instance of this genome with a mutated DNA and model"""

        parents = [self.dna]
        dna = self.dna.mutate()
        model = copy.deepcopy(self.model)
        mutation_min = 1/mutation_amount + epsilon # add epsilon to avoid weight decay to always push the weights to 0, only should happen for higher values
        mutation_multiplier = mutation_amount - mutation_min
        # mutate the model
        for param_old, param_new in zip(self.model.parameters(), model.parameters()):
            chances = torch.rand(param_old.shape)
            mask = chances < mutation_rate
            mutations = (torch.randn(param_old.shape)) * mutation_amount
            #my_rand = torch.rand(param_old.shape)
            #mutations = (my_rand*mutation_multiplier+mutation_min)

            if weight_decay:
                # center weights around 0
                # this might not be the most efficient way to do this
                mutations /= torch.exp(torch.pow(param_old/20, 2))


            # add epsilon to avoid multiplying by 0 and getting stuck
            # also makes it possible to reverse sign of mutation
            new_data = param_old * mutations + mutations * epsilon
            param_new.data = torch.where(mask, new_data, param_old)
        return Genome(model, parents, dna, population=self.population)
    
    def crossover(self, other:"Genome")->"Genome":
        """Returns a new instance of this genome with a crossover of the DNA and model"""
        parents = [self.dna, other.dna]
        dna = self.dna.crossover(other.dna)
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

    def __init__(self, factory: callable, fitness_function: callable, inverse_fitness=False, size: int = 128, weights: dict = None, device=None):
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
        #self.top_weighting = lambda idx: 1 / 1 + idx # weighting of the top genomes
        self.top_weighting = lambda idx: 1
        self.generations = [] # old generations, usually without the models
        self.fitness_history = []
        self.genomes = []
        self.verbose = True # print progress
        self.weights = {
            "mutation": 20,
            "crossover": 0,
            "elitism": 2,
            "random": 10
        }
        if weights is not None:
            self.weights.update(weights)

        self.device = device

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

        #self.generations.append(old_genomes)


    def fill_population(self):
        """Fill the population with new genomes"""
        with torch.device(self.device) if self.device is not None else NoWith():
            with torch.no_grad():
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
        with torch.device(self.device) if self.device is not None else NoWith():
            with torch.no_grad():
                for genome in self.genomes:
                    genome.fitness = self.fitness_function(genome)
    
    def evolve(self, generations=1):

        with torch.device(self.device) if self.device is not None else NoWith():
            with torch.no_grad():
                if self.generation == 0:
                    #self._print(f"Evaluating generation {self.generation}")
                    # evaluate the fitness of the genomes
                    all_fitness = []
                    for genome in tqdm.tqdm(self.genomes):
                        fitness = self.fitness_function(genome)
                        genome.fitness = fitness
                        all_fitness.append(fitness)
                    self.fitness_history.append(all_fitness)

                best_fit = self.get_best_genome().fitness
                bf_s = f", Top: {best_fit:.2f}"

                for _ in range(generations):
                    
                    self._print(f"Creating generation {self.generation+1}")
                    # create the next generation
                    self.next_generation()

                    #self._print(f"Evaluating generation {self.generation}")
                    # evaluate the fitness of the genomes
                    all_fitness = []
                    for genome in tqdm.tqdm(self.genomes):
                        fitness = self.fitness_function(genome)
                        genome.fitness = fitness
                        all_fitness.append(fitness)
                    self.fitness_history.append(all_fitness)

                    # sort the genomes by fitness
                    self.genomes.sort(key=lambda g: g.fitness, reverse=(not self.inverse_fitness))
                    # print best 5 fitness
                    g1 = self.genomes[0]
                    g2 = self.genomes[1]
                    dna_similarity = g1.dna.similarity(g2.dna)
                    print("Top similarity:", round(dna_similarity*100, 2))
                    self._print(f"Best fitness: {[round(float(g.fitness), 3) for g in self.genomes[:5]]}")
                    best_fit = self.get_best_genome().fitness
                    bf_s = f", Top: {best_fit:.2f}"

    def plot_fitness(self):
        plt.plot([g.fitness for g in self.genomes])
        plt.show()

    def plot_fitness_history(self, save=False, name=None):
        # plot best and avg fitness in each generation
        if name is None:
            name = "Fitness History"
        best_fitness = [max(g) for g in self.fitness_history]
        avg_fitness = [sum(g)/len(g) for g in self.fitness_history]
        fig = plt.figure()
        plt.plot(best_fitness, label="Best")
        plt.plot(avg_fitness, label="Average")
        plt.legend()
        plt.title(name)
        if save:
            fig.savefig(name+".png")
        else:

            plt.show()
        

    def get_best_model(self):
        return self.genomes[0].model
    
    def get_best_genome(self):
        return self.genomes[0]


        
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