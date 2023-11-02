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
