from src.evolve import Population

import torch
import torch.nn as nn

import matplotlib.pyplot as plt


class XOR(nn.Module):
    def __init__(self):
        super(XOR, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.fc1(x))
        x = self.fc2(x)
        return x
x, y = None, None

x = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float)
loss = nn.MSELoss()

# abs err
loss2 = nn.L1Loss()

def fitness(genome):
    y_pred = genome.model(x)
    
    # calculate loss
    l = loss(y_pred, y)
    return 1 / (l + 1e-6)


def testXOR():
    pop = Population(lambda: XOR(), fitness, size=128)
    pop.verbose = True
    pop.evolve(1000)

    ev_model = pop.get_best_model()
    print(ev_model(x))

    losses = []

    # train with gradient descent
    model = XOR()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    for _ in range(200):
        y_pred = model(x)
        l = loss(y_pred, y)
        l.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(l.item())

    print(model(x))

    # save loss history
    plt.plot(losses)
    plt.savefig("plots/XOR_loss_200e.png")

    # print weights
    print(ev_model.fc1.weight)
    print(model.fc1.weight)

    pop.plot_fitness_history(save=True, name="plots/XOR_test_1000g")

    # test

    y_gd = model(x)
    loss_gd = loss(y_gd, y).item()
    y_ev = ev_model(x)
    loss_ev = loss(y_ev, y).item()

    print("XOR True")
    print(y)
    print("GD")
    print("Pred", y_gd)
    print("Loss", loss_gd)
    print("EV")
    print("Pred", y_ev)
    print("Loss", loss_ev)




