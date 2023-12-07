import torch
from torch import nn

from src.evolve import Population
from src.fitness import StochasticFitness
from src.util import MutateModelFactory

import testing.birdgame.birdgame as birdgame
import testing.birdgame.main as game_main

class Finch(birdgame.Bird):

    def __init__(self, game=None, model=None):
        super().__init__(game)
        
        self.model = model
        self.print_debug = False


    def get_bird_state(self):
        """
        returns values that represent the state of the bird and the game that can be fed into the model

        Returns:
            torch.tensor: tensor of shape (9,) representing the state of the bird and the game
        """
        y = self.y
        y_vel = self.y_velocity
        x = self.score # score is the distance traveled

        next_pipes = [] # stores the next pipes
        max_pipes = 2 # number of pipes to store
        for i in range(len(self.game.pipes)):
            pipe = self.game.pipes[i]
            # check if pipe is in front of bird
            pipe_end = pipe.x + pipe.width
            if pipe_end > x:
                next_pipes.append(pipe)
            
            if len(next_pipes) == max_pipes:
                break

        pipe_1_dist = 999
        pipe_1_gap_y = 0.5
        pipe_1_gap_size = 0.5

        pipe_2_dist = 999
        pipe_2_gap_y = 0.5
        pipe_2_gap_size = 0.5

        if len(next_pipes) > 0:
            pipe_1 = next_pipes[0]
            pipe_1_dist = pipe_1.x - x
            pipe_1_gap_y = pipe_1.gap_y
            pipe_1_gap_size = pipe_1.gap_size

        if self.print_debug:
            print("pipe_1_dist", pipe_1_dist)
            print("pipe_1_gap_y", pipe_1_gap_y)
            print("pipe_1_gap_size", pipe_1_gap_size)

        if len(next_pipes) > 1:
            pipe_2 = next_pipes[1]
            pipe_2_dist = pipe_2.x - x
            pipe_2_gap_y = pipe_2.gap_y
            pipe_2_gap_size = pipe_2.gap_size

        return torch.tensor([y, y_vel, x, pipe_1_dist, pipe_1_gap_y, pipe_1_gap_size])

    def tick(self, dt):
        # tick function is called every frame
        # we use the model to predict whether to jump or not

        # get the state of the bird and the game
        state = self.get_bird_state()

        # feed the state into the model
        output = self.model(state)
        # the model returns a value between 0 and 1
        # if the value is greater than 0.5, the bird jumps
        if output > 0.5:
            self.jump()


class BirdModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(6, 16)
        #self.layer2 = nn.Linear(16, 16)
        self.layer3 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        #x = self.layer2(x)
        #x = self.relu(x)
        x = self.layer3(x)
        x = self.sigmoid(x)
        return x
    
game = birdgame.Birdgame()
game.verbose = False
    
def test_fitness(genome):
    model = genome.get_model()
    bird = Finch(model=model)
    
    fitness = game.get_fitness(bird, runs=2)

    return fitness
    
    
def test_birdgame(gen=200, name=None):

    # next: use saved model and MutateModelFactory to continue training
    # load model
    model = BirdModel()
    model.load_state_dict(torch.load("best_bird.pt"))
    from src.util import MutateModelFactory
    mmf = MutateModelFactory(model)

    pop = Population(mmf, test_fitness, size=100)
    pop.verbose = True
    pop.evolve(gen)

    best_model = pop.get_best_model()
    # save the best model
    torch.save(best_model.state_dict(), "best_bird.pt")

    # plot hist
    if name is None:
        name = f"Bird test {gen} generations"
    name = "plots/" + name
    pop.plot_fitness_history(name=name, save=True)

def run_game():
    model = BirdModel()
    model.load_state_dict(torch.load("best_bird.pt"))
    bird = Finch(model=model)
    #bird.print_debug = True	

    game_main.main([birdgame.Bird(), bird])

    


