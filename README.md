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
Genetic Algorithms offer a unique approach to neural network training, presenting both advantages and disadvantages compared to traditional methods like gradient descent. They are not an all-around solution, but can be worth considering if you problem can take advantage of their benefits.

### Advantages
#### Global Search
GAs are effective for exploring a broad solution space, making them well-suited for complex optimization problems where the optimal solution is not easily determined.

#### Data-Efficient Learning with Genetic Algorithms
Genetic Algorithms (GAs) offer a distinct advantage in that they don't necessitate extensive labeled training data. Unlike conventional machine learning methods, GAs operate on an evolutionary paradigm, making them particularly suitable for scenarios where gathering large datasets is challenging or impractical.

#### No Gradient Dependencies
Unlike gradient-based methods, GAs do not rely on the availability of gradients, making them applicable to non-differentiable and discontinuous objective functions.

#### Parallel Exploration
GAs can naturally explore multiple potential solutions in parallel through the population-based evolution process, potentially leading to faster convergence.

#### Combination with other methods
Genetic Algorithms can be combined with other methods like gradient descent. They can be used to improve a pretrained model by using it as the initial population, potentially resulting in better performance and better generelization capabilities.

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

Explore the "testing" directory for practical implementations of genetic algorithms on some simple optimization problems.


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

### Example Implementations
There are a few implemented examples that show some of the features in the testing directory. You can look at them directly or try to run the test.py file to see how it works.

#### XOR Test
In this test we try to train a simple network to emulate an XOR Gate by receiving two inputs, 1 or 0, and turning them into a single output.

To train the network, we create two tensors of all possible inputs and their correct outputs. To evaluate the fitness, we simply use a loss function to calculate the average loss for the samples and return the inverse, since we want a higher fitness value for better results.

After 1000 generations, we select the best performing model and measure its performance.
![XOR_test_1000gen](https://github.com/4-en/EvolveTorch/assets/105049118/4451c2db-2375-4c47-9e2e-d73daa9c8cec)

#### Cosine test
In this test we train the network to output the cosine of its input. Like previously, we use some example inputs as well as their correct outputs and a loss function to calculate the fitness of each genome.
Since this model is a lot bigger than the previous one, it takes longer to train. Still, we can see some results after 50 generations, although training for longer would improve it further.
![cosine_test_50gen](https://github.com/4-en/EvolveTorch/assets/105049118/441406b7-0c62-4385-a88a-c9aae46d2c80)

#### Game AI test
In this test, we try to train an AI that plays a simple game. In the game, the player has to navigate his character between obstacles in a endless level that is randomly generated. The longer the player survives without crashing, the higher his score gets.

There is only one input, space, which makes the player jump until gravity catches up and he has to jump again. We will train a model that takes multiple inputs like score, position, velocity and position of the next obstacle to output a single value. If that value is above 0.5, the jump key is pressed.

The interesting part in this example is, that we don't know what the optimal solution is. We don't know when the player should press space for every input and therefore we don't have a dataset we could just use to train with gradient descent like in the previous examples. Instead, to determine the fitness, we will just run the game with every model until the player crashes. The score will be equal to the fitness, which means that models that reach a higher score will be used to create the next generation and improve the performance further.

![Birdgame 100pop 500gen](https://github.com/4-en/EvolveTorch/assets/105049118/9ee9b000-a8ba-491d-9739-0fd534c88e78)

The graph shows that the algorithm keeps improving the models over time, even if it takes relatively long. After 500 generations, the model can consistently reach high scores.

## Roadmap
This roadmap shows some of the planned or possible features. While there is always more that could be improved, it serves as a general outline for the project.
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
-  - [x] Test different strategies and values
-  - [x] Test performance in different scenarios
-  - [ ] Compare with other methods like supervised learning and reinforcement learning

### Main Evolution Loop
- [x] Create a loop that iteratively evolves the population over multiple generations.
- [x] Decide on a termination condition (e.g., maximum generations or fitness threshold).

### Logging and Visualization
- [x] Implement logging mechanisms to track the progress of the evolutionary process.
- [x] Explore visualization tools for population statistics, fitness values, and convergence analysis.

### Performance Optimization
- [ ] improve performance if possibly :)

### Further applications and ideas
- [ ] use evolution to improve trained model after local minimum has been reached
