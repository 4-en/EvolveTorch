# this file contains functions related to the fitness of the population

import torch.nn as nn

class FitnessFromDataset:
    """
    calculates the fitness of the model based on the dataset
    uses a loss function to calculate the fitness
    """
    def __init__(self, dataset, loss_f=None, inverse_loss=True):
        self.dataset = dataset

        self.loss_f = nn.MSELoss() if loss_f is None else loss_f
        self.inverse_loss = inverse_loss

    def fitness(self, model):
        """
        calculates the fitness of the model based on the dataset
        """
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
    
    def __call__(self, model):
        return self.fitness(model)
    