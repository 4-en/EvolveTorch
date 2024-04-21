# this is the code for the game without rendering

import random

class Pipe:
    def __init__(self, x, gap_y, width=0.2, gap_size=0.3):
        self.x = x
        self.gap_y = gap_y # bottom of the gap
        self.width = width
        self.gap_size = gap_size


    def check_collision(self, bird):
        if bird.score + bird.size < self.x:
            return False
        if bird.score > self.x + self.width:
            return False
        if bird.y < self.gap_y:
            #print("bird below gap")
            #print("bird.y", bird.y, "gap_y", self.gap_y)
            return True
        if bird.y + bird.size > self.gap_y + self.gap_size:
            #print("bird above gap")
            #print("bird.y", bird.y, "gap_y", self.gap_y)
            return True
        return False

class Bird:
    def __init__(self, game=None):
        self.score = 0 # equal to distance traveled
        self.y = 0.5 # y position of the bird, height
        self.y_velocity = 0 # y velocity of the bird
        self.jump_cooldown = 0
        self.dead = False
        self.size = 0.03
        self.game = game

    def reset(self):
        self.score = 0
        self.y = 0.5
        self.y_velocity = 0
        self.jump_cooldown = 0
        self.dead = False

    def jump(self):
        if self.jump_cooldown <= 0:
            self.y_velocity = self.y_velocity * 0.5 + 0.5
            self.jump_cooldown = 0.2

    def tick(self, dt):
        pass

    
    
    def check_collision(self, pipes):
        for pipe in pipes:
            if pipe.check_collision(self):
                self.dead = True
                return True
        return False
    
class AIBird(Bird):

    def tick(self, dt):
        # simple ai
        # get next pipe
        next_pipe = None
        for pipe in self.game.pipes:
            if pipe.x + pipe.width > self.score:
                next_pipe = pipe
                break
        if next_pipe is None:
            return
        # jump if gap is above
        if self.y - 2* self.size  < next_pipe.gap_y:
            self.jump()
        return

class Birdgame:

    def  __init__(self):
        self.birds = [Bird(self)] # future: multiple birds for genetic algorithm
        self.pipes = []

        self.score = 0 # equal to distance traveled, if only one bird, this is the bird's score

        self.reset()
        self.speed = 0.4
        self.gravity = -0.7
        self.next_pipe = 0

    def reset(self):
        #self.birds = [Bird(self), AIBird(self)] # future: multiple birds for genetic algorithm
        for bird in self.birds:
            bird.reset()
        self.pipes = []
        self.score = 0
        self.speed = 0.4
        self.next_pipe = 0

    def game_over(self):
        for bird in self.birds:
            if not bird.dead:
                return False
        return True
    
    def spawn_pipe(self):
        if self.score+2 > self.next_pipe:
            difficulty = min(1, self.score/20)
            gap_size = 0.3 - 0.1*difficulty
            width = 0.2
            gap_y = random.uniform(0.05, 0.95-gap_size)
            self.pipes.append(Pipe(self.score + 2, gap_y, width, gap_size))
            #print("spawned pipe at", self.score + 2)
            self.next_pipe = self.score + 2 + random.uniform(1.2-(0.4*difficulty), 2.5-1*difficulty)

    def clean_pipes(self):
        for pipe in self.pipes:
            if pipe.x < self.score - 10:
                self.pipes.remove(pipe)
        

    def tick(self, dt=1/60):
        if self.game_over():
            return
        

        movement = self.speed * dt #* (1 + min(1.5, self.score/30))
        self.score += movement

        self.spawn_pipe()
        self.clean_pipes()

        for bird in self.birds:
            if not bird.dead:
                bird.tick(dt)
            bird.y_velocity += self.gravity * dt
            bird.y += bird.y_velocity * dt

            if bird.y < 0 or bird.y > 1:
                bird.dead = True
            
            if bird.y < 0:
                bird.y = 0

            if bird.dead:
                continue

            if bird.jump_cooldown > 0:
                bird.jump_cooldown -= dt

            if bird.check_collision(self.pipes):
                bird.dead = True

            bird.score = self.score

    def get_fitness(self, bird, runs=1, max_score=9999):
        """
        Runs the game for a bird and returns the fitness
        Only works for AI controlled birds
        """
        bird.game = self
        score = 0
        for _ in range(runs):
            self.reset()
            bird.reset()
            self.birds = [bird]
            while not self.game_over():
                self.tick()
                if self.score > max_score:
                    break

            score += self.score
                
        
        return score / runs
    

if __name__ == "__main__":
    game = Birdgame()
    bird = AIBird()

    score = game.get_fitness(bird, 10)
    print("Fitness of AIBird: ", score)