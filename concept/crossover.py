# test crossover in pytorch nn

import torch
import torch.nn as nn

class Crossover(nn.Module):
    def __init__(self, weight_value=0.5):
        super(Crossover, self).__init__()
        self.weight_value = weight_value

        self.layer1 = nn.Linear(10, 10)
        self.layer2 = nn.Linear(10, 10)

        # set all weights to weight_value
        for param in self.parameters():
            param.data = torch.ones(param.data.shape) * weight_value

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x
    
    def crossover(self, other):
        for param1, param2 in zip(self.parameters(), other.parameters()):
            rand = torch.rand(param1.data.shape)
            param1.data = torch.where(rand < self.weight_value, param1.data, param2.data)
        return self
    
    def print_params(self):
        for param in self.parameters():
            print(param.data)

def main():
    net1 = Crossover(weight_value=0.5)
    net2 = Crossover(weight_value=-0.5)
    print("Net1:")
    net1.print_params()
    print("Net2:")
    net2.print_params()
    print("Crossover:")
    net1.crossover(net2)
    net1.print_params()

if __name__ == "__main__":
    main()

    # note: seems to work fine :)