import pygame
import sys
import random

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
BIRD_WIDTH = 40
BIRD_HEIGHT = 30
PIPE_WIDTH = 50
GAP = 200
PIPE_SPEED = 3
GRAVITY = 0.25
JUMP_STRENGTH = 6

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)

# Create the screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Flappy Bird Clone")

# Bird
bird_x = 50
bird_y = SCREEN_HEIGHT // 2
bird_velocity = 0

# Pipes
pipes = []
pipe_x = SCREEN_WIDTH

def create_pipe():
    pipe_height = random.randint(50, SCREEN_HEIGHT - GAP - 50)
    pipes.append([pipe_x, 0, PIPE_WIDTH, pipe_height])
    pipes.append([pipe_x, pipe_height + GAP, PIPE_WIDTH, SCREEN_HEIGHT - pipe_height - GAP])

def draw_bird(x, y):
    pygame.draw.rect(screen, GREEN, (x, y, BIRD_WIDTH, BIRD_HEIGHT))

def draw_pipes():
    for pipe in pipes:
        pygame.draw.rect(screen, GREEN, pipe)

def check_collision():
    for pipe in pipes:
        if bird_x + BIRD_WIDTH > pipe[0] and bird_x < pipe[0] + PIPE_WIDTH:
            if bird_y < pipe[1] or bird_y + BIRD_HEIGHT > pipe[1] + GAP:
                return True
    if bird_y < 0 or bird_y + BIRD_HEIGHT > SCREEN_HEIGHT:
        return True
    return False

# Main game loop
create_pipe()
clock = pygame.time.Clock()
score = 0

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                bird_velocity = -JUMP_STRENGTH

    bird_y += bird_velocity
    bird_velocity += GRAVITY

    if pipes and pipes[0][0] < -PIPE_WIDTH:
        pipes.pop(0)
        pipes.pop(0)

    if pipes and pipes[-1][0] < SCREEN_WIDTH - PIPE_WIDTH * 2:
        create_pipe()
        score += 1

    for pipe in pipes:
        pipe[0] -= PIPE_SPEED

    screen.fill(WHITE)
    draw_bird(bird_x, bird_y)
    draw_pipes()

    if check_collision():
        break

    pygame.display.update()
    clock.tick(30)

# Game over screen
font = pygame.font.Font(None, 36)
game_over_text = font.render("Game Over", True, GREEN)
screen.blit(game_over_text, (SCREEN_WIDTH // 2 - 100, SCREEN_HEIGHT // 2 - 18))
score_text = font.render("Score: " + str(score), True, GREEN)
screen.blit(score_text, (SCREEN_WIDTH // 2 - 60, SCREEN_HEIGHT // 2 + 18))
pygame.display.update()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    pygame.display.update()
