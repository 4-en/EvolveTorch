# EvolveTorch
A simple genetic algorithm library for PyTorch
## Introduction
This Python library is designed to implement and experiment with genetic algorithms and evolutionary algorithms using the PyTorch framework. Genetic algorithms draw inspiration from the process of natural selection to optimize solutions to complex problems.

## What are Genetic Algorithms?
Genetic algorithms are optimization algorithms inspired by the process of natural selection. They are used to find approximate solutions to optimization and search problems. In a genetic algorithm, a population of candidate solutions undergoes iterative generations of selection, crossover, and mutation to evolve towards optimal or near-optimal solutions.

### Key Components of Genetic Algorithms
- Population: A set of potential solutions to the optimization problem.
- Fitness Function: A function that assigns a numerical score to each solution based on how well it satisfies the optimization criteria.
- Selection: Individuals are chosen from the population based on their fitness scores to serve as parents for the next generation.
- Crossover: Pairs of parents combine their genetic information to create offspring.
- Mutation: Random changes are introduced to individual solutions to maintain diversity.
- Termination Criteria: The algorithm stops when a specified condition is met (e.g., a maximum number of generations or satisfactory fitness level).

## Goal of Genetic Algorithms:
The primary goal of genetic algorithms is to efficiently explore a solution space and converge towards optimal or near-optimal solutions. By mimicking the principles of natural selection, genetic algorithms can be applied to a wide range of optimization problems, including parameter tuning in machine learning models, scheduling, and routing problems.

## Why Genetic Algorithms?
Genetic Algorithms offer a unique approach to neural network training, presenting both advantages and disadvantages compared to traditional methods like gradient descent.

### Advantages
#### Global Search
GAs are effective for exploring a broad solution space, making them well-suited for complex optimization problems where the optimal solution is not easily determined.

#### No Gradient Dependencies
Unlike gradient-based methods, GAs do not rely on the availability of gradients, making them applicable to non-differentiable and discontinuous objective functions.

#### Parallel Exploration
GAs can naturally explore multiple potential solutions in parallel through the population-based evolution process, potentially leading to faster convergence.

### Disadvantages
#### Computational Intensity
Training neural networks using GAs can be computationally intensive, especially for large and complex networks, making them less efficient compared to gradient-based optimization methods.

#### Search Space Explosion
In highly complex optimization spaces, the number of possible solutions grows exponentially. GAs may struggle to efficiently explore and exploit this vast solution space, leading to suboptimal convergence.

#### Risk of Premature Convergence
GAs can be prone to premature convergence, where the population converges to a suboptimal solution before exploring the full solution space, particularly in complex optimization landscapes.

#### Limited Understanding of Inner Workings
GAs provide a black-box optimization approach, making it challenging to gain insights into the internal workings of the evolved neural networks.


## Getting Started
Follow the examples and documentation provided in the library to apply genetic algorithms to your optimization problems. Experiment with different parameters, mutation strategies, and crossover methods to tailor the algorithm to your specific use case.

Explore the "testing" directory for practical implementations of genetic algorithms on various optimization problems.


### Workflow
- create PyTorch model to train with EvolveTorch
- create fitness function for the specific model
- create EvolveTorch instance with model and fitness function
- specify parameters (if using non default values)
- - population size (amount of different models in each generation)
  - elitism rate (% of top models to keep unchanged)
  - mutation properties
  - crossover properties
- train for n generations (or until some fitness goal is met)

### Example
```python
# define your model architecture
class MyModel(nn.Module):
  ...

# define a fitness function or use predefined from fitness.py
def my_fitness(genome):
  model = genome.get_model()
  # x and y are input and output values from a dataset
  y_pred = model(x)
  # calculate loss
  l = loss(y_pred, y)

  # higher fitness -> better
  return 1 / (l + 1e-6)

# create Population
pop = Population(lambda: MyModel(), my_fitness, size=128)

# evolve for n generations
n = 1000
pop.evolve(n)

# get best model and use it / save weights
best_model = pop.get_best_model()
...

```

## Roadmap
### Concept
- [x] test basic ideas in small tests

### Core Functions
-  Population Initialization
-  - [x] Create functions to generate an initial population of neural networks.
-  - [x] Randomly initialize the weights for predefined network architecture.

- Fitness Evaluation
-  - [x] Create mechanism to compare fitness of networks using given fitness function.

- Selection
-  - [x] Develop selection mechanisms to choose the top-performing networks as parents for the next generation.

### Evolutionary Operations
- Mutation
-  - [x] Create mutation functions to introduce small random changes to the weights of neural networks.
-  - [ ] Experiment with different mutation strategies (e.g., Gaussian perturbation).

- Crossover (Recombination)
-  - [x] Implement crossover functions to combine genetic information from two parent networks to create offspring.
-  - [ ] Explore various crossover strategies (e.g., one-point or uniform crossover).

- Population Replacement
-  - [x] Define how new networks will replace the old generation, considering elitism and diversity.
 
- DNA mechanism
-  - [x] implement a mechanism that represents DNA to avoid crossover between certain models
-  - [ ] avoid crossover between too different models (-> prevent useless combinations since likelyhood would be very low if models are too different)
-  - [ ] avoid crossover between too similar models (-> prevents creating models that are too similar, which would lead to less/no progress)
-  - [ ] remove models with too similar dna from population to maximise new combinations

- Evolution Optimization
-  - [ ] Test different strategies and values
-  - [ ] Test performance in different scenarios
-  - [ ] Compare with other methods like supervised learning and reinforcement learning

### Main Evolution Loop
- [x] Create a loop that iteratively evolves the population over multiple generations.
- [ ] Decide on a termination condition (e.g., maximum generations or fitness threshold).

### Logging and Visualization
- [ ] Implement logging mechanisms to track the progress of the evolutionary process.
- [ ] Explore visualization tools for population statistics, fitness values, and convergence analysis.

### Performance Optimization
- [ ] improve performance if possibly :)

### Further applications and ideas
- [ ] use evolution to improve trained model after local minimum has been reached
