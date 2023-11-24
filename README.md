# EvolveTorch
a simple framework to apply evolutionary mechanisms to PyTorch networks

## Goals
- provide different functions to train networks using genetic algorithm
- simple and easy to use

## Workflow (wip)
- create model to train with EvolveTorch
- create fitness function for the specific model
- create EvolveTorch instance with model and fitness function
- specify parameters (if using non default values)
- - population size (amount of different models in each generation)
  - elitism rate (% of top models to keep unchanged)
  - mutation properties
  - crossover properties
- train for n generations (or until some goal is met)

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
