# this is the code for the game without rendering

import random

class Pipe:
    def __init__(self, x, gap, width=50):
        self.x = x
        self.gap = gap
        self.width = width


    def check_collision(self, bird):
        if bird.score + bird.size < self.x:
            return False
        if bird.score > self.x + self.width:
            return False
        if bird.y < self.gap - bird.size/2:
            return True
        if bird.y > self.gap + bird.size/2:
            return True
        return False

class Bird:
    def __init__(self):
        self.score = 0 # equal to distance traveled
        self.y = 0.5 # y position of the bird, height
        self.y_velocity = 0 # y velocity of the bird
        self.jump_cooldown = 0
        self.dead = False
        self.size = 0.03

    def reset(self):
        self.score = 0
        self.y = 0.5
        self.y_velocity = 0
        self.jump_cooldown = 0
        self.dead = False

    def jump(self):
        if self.jump_cooldown <= 0:
            self.y_velocity = -0.5
            self.jump_cooldown = 0.2

    def tick(self, dt):
        return
    
    def check_collision(self, pipes):
        for pipe in pipes:
            if pipe.check_collision(self):
                self.dead = True
                return True
        return False

class Birdgame:

    def  __init__(self):
        self.birds = [] 
        self.pipes = []

        self.score = 0 # equal to distance traveled, if only one bird, this is the bird's score

        self.reset()
        self.speed = 0.5
        self.gravity = 0.6
        self.next_pipe = 0

    def reset(self):
        self.birds = [Bird()] # future: multiple birds for genetic algorithm
        self.pipes = []
        self.score = 0
        self.speed = 1
        self.next_pipe = 0

    def game_over(self):
        for bird in self.birds:
            if not bird.dead:
                return False
        return True
    
    def spawn_pipe(self):
        if self.score+3 > self.next_pipe:
            self.pipes.append(Pipe(self.score + 3, random.uniform(0.2, 0.8)))
            self.next_pipe = self.score + 3 + random.uniform(1, 3)

    def clean_pipes(self):
        for pipe in self.pipes:
            if pipe.x < self.score - 10:
                self.pipes.remove(pipe)
        

    def tick(self, dt=1/60):
        if self.game_over():
            return
        
        movement = self.speed * dt
        self.score += movement

        self.spawn_pipe()
        self.clean_pipes()

        for bird in self.birds:
            if bird.dead:
                continue
            bird.tick(dt)
            bird.y_velocity += self.gravity * dt
            bird.y += bird.y_velocity * dt

            if bird.y < 0 or bird.y > 1:
                bird.dead = True

            if bird.jump_cooldown > 0:
                bird.jump_cooldown -= dt

            if bird.check_collision(self.pipes):
                bird.dead = True

            bird.score = self.score

        