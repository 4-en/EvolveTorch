# this file contains functions related to the fitness of the population

import torch.nn as nn
import torch
import numpy as np

class FitnessFromDataset:
    """
    calculates the fitness of the model based on the dataset
    uses a loss function to calculate the fitness
    """
    def __init__(self, dataset, loss_f=None, inverse_loss=True):
        self.dataset = dataset

        self.loss_f = nn.MSELoss() if loss_f is None else loss_f
        self.inverse_loss = inverse_loss

    def fitness(self, genome):
        """
        calculates the fitness of the model based on the dataset
        """
        model = genome.model
        model.eval()
        total_loss = 0
        for x, y in self.dataset:
            y_pred = model(x)
            loss = self.loss_f(y_pred, y)
            total_loss += loss.item()
        avg_loss = total_loss / len(self.dataset)
        if self.inverse_loss:
            return 1 / avg_loss + 1e-9
        return avg_loss
    
    def __call__(self, genome):
        return self.fitness(genome)
    
class StochasticFitness:
    """
    def calculates the fitness of the model based on the dataset
    only uses a subset of the dataset
    """

    def __init__(self, dataset, loss_f=None, inverse_loss=True, subset_size=100):
        self.dataset = dataset
        self.subset_size = subset_size

        self.loss_f = nn.MSELoss() if loss_f is None else loss_f
        self.inverse_loss = inverse_loss

    def get_subset(self):
        """
        returns a subset of the dataset
        """
        if self.subset_size >= len(self.dataset) and False:
            # TODO: this is not implemented correctly, pls fix
            return self.dataset
        return torch.utils.data.Subset(self.dataset, np.random.choice(len(self.dataset), self.subset_size))
    
    def fitness(self, genome):

        model = genome.model
        subset = self.get_subset()
        model.eval()

        total_loss = 0
        for x, y in subset:
            y_pred = model(x)
            loss = self.loss_f(y_pred, y)
            total_loss += loss.item()

        avg_loss = total_loss / len(subset)
        if self.inverse_loss:
            return 1 / avg_loss + 1e-9
        
        return avg_loss
    
    def __call__(self, genome):
        return self.fitness(genome)
