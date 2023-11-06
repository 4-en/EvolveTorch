# EvolveTorch
a simple framework to apply evolutionary mechanisms to PyTorch networks

## Goals
- provide different functions to train networks using different evolutionary techniques
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
- [ ] test basic ideas in small tests

### Core Functions
-  Population Initialization
-  - [ ] Create functions to generate an initial population of neural networks.
-  - [ ] Randomly initialize the weights for predefined network architecture.

- Fitness Evaluation
-  - [ ] Create mechanism to compare fitness of networks using given fitness function.

- Selection
-  - [ ] Develop selection mechanisms to choose the top-performing networks as parents for the next generation.

### Evolutionary Operations
- Mutation
-  - [ ] Create mutation functions to introduce small random changes to the weights of neural networks.
-  - [ ] Experiment with different mutation strategies (e.g., Gaussian perturbation).

- Crossover (Recombination)
-  - [ ] Implement crossover functions to combine genetic information from two parent networks to create offspring.
-  - [ ] Explore various crossover strategies (e.g., one-point or uniform crossover).

- Population Replacement
-  - [ ] Define how new networks will replace the old generation, considering elitism and diversity.

### Main Evolution Loop
- [ ] Create a loop that iteratively evolves the population over multiple generations.
- [ ] Decide on a termination condition (e.g., maximum generations or fitness threshold).

### Logging and Visualization
- [ ] Implement logging mechanisms to track the progress of the evolutionary process.
- [ ] Explore visualization tools for population statistics, fitness values, and convergence analysis.

### Performance Optimization
- [ ] improve performance if possibly :)
