import torch
from torch import nn

from src.evolve import Population
from src.fitness import StochasticFitness
from src.util import MutateModelFactory
from math import pi as PI


class CosineModule(nn.Module):
    def __init__(self):
        super(CosineModule, self).__init__()
        self.layer1 = nn.Linear(1, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, 1)
        self.ReLU = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        x = self.ReLU(x)
        x = self.layer2(x)
        x = self.ReLU(x)
        x = self.layer3(x)
        #x = self.sigmoid(x)
        return x
    
    

def cosine_test():
    x = torch.arange(0, 2*3.1415926, 0.01)
    x = x.unsqueeze(1)
    y = torch.cos(x)

    model = CosineModule()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    EPOCHS = 1000
    for epoch in range(EPOCHS):
        x = torch.rand(1000, 1) * 2 * PI
        y = torch.cos(x)
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        print('epoch {}, loss {}'.format(epoch, loss.item()))

    pop = Population(MutateModelFactory(model), StochasticFitness([(x,y)]), size=128)
    pop.verbose = True
    pop.evolve(200)

    ev_model = pop.get_best_model()
    test_values = torch.rand(1000) * 2 * PI
    test_values = test_values.unsqueeze(1)

    ev_model.eval()
    model.eval()
    y_true = torch.cos(test_values)
    y_sl = model(test_values)
    y_ev = ev_model(test_values)

    # avg error
    print("avg error")
    print("Supervised:")
    print(torch.mean(torch.abs(y_true - y_sl)))
    print("Evolved:")
    print(torch.mean(torch.abs(y_true - y_ev)))

    # max error
    print("max error")
    print("Supervised:")
    print(torch.max(torch.abs(y_true - y_sl)))
    print("Evolved:")
    print(torch.max(torch.abs(y_true - y_ev)))





    
