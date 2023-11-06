# testing mutation of pytorch networks


import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
    

def mutate(network:nn.Module, mutation_rate:float, mutation_strength:float=0.1) -> nn.Module:
    for param in network.parameters():
        if torch.rand(1) < mutation_rate:
            param.data += torch.randn(param.data.shape) * mutation_strength
    return network

def print_params(network:nn.Module):
    for param in network.parameters():
        print(param.data)

def main():
    net = Net()
    print_params(net)
    print("Mutating...")
    net = mutate(net, 0.5, 0.1)
    print_params(net)

if __name__ == "__main__":
    main()