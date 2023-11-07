import pygame
import sys
import random

import birdgame




# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
FPS = 60

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
SKY_BLUE = (135, 206, 235)
ORANGE = (255, 165, 0)
BLACK = (0, 0, 0)

def generate_bird_color():
    color = SKY_BLUE
    while sum([abs(color[i] - SKY_BLUE[i]) for i in range(3)]) < 100:
        # generate a color that is not too close to sky blue
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    return color


def main():
    # Initialize pygame
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Bird Game")

    # Initialize game
    game = birdgame.Birdgame()

    bird_pos = SCREEN_WIDTH // 3
    wait_time = 1 / FPS
    
    # create font
    font = pygame.font.Font(None, 36)

    # Game loop
    while True:
        jump = False
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if not game.game_over():
                        jump = True
                    else:
                        game.reset()
            

        if jump:
            if len(game.birds) == 1 and not game.birds[0].dead:
                game.birds[0].jump()
        # Game logic
        game.tick(wait_time)

        score = int(game.score*10)

        current_zero = bird_pos + game.score * SCREEN_HEIGHT

        # Draw
        screen.fill(SKY_BLUE)
        for pipe in game.pipes:
            pygame.draw.rect(screen, GREEN, pygame.Rect(pipe.x * SCREEN_HEIGHT - current_zero, 0, 0.1 * SCREEN_HEIGHT, pipe.gap * SCREEN_HEIGHT))
            pygame.draw.rect(screen, GREEN, pygame.Rect(pipe.x * SCREEN_HEIGHT - current_zero, pipe.gap * SCREEN_HEIGHT + 0.2 * SCREEN_HEIGHT, 0.1 * SCREEN_HEIGHT, SCREEN_HEIGHT))
        for bird in game.birds:
            pygame.draw.rect(screen, BLACK, pygame.Rect(bird_pos, bird.y * SCREEN_HEIGHT, bird.size * SCREEN_HEIGHT, bird.size * SCREEN_HEIGHT))
        

        # draw score on top right

        text = font.render(str(score), True, BLACK)
        textpos = text.get_rect()
        textpos.centerx = SCREEN_WIDTH - 50
        textpos.centery = 50
        screen.blit(text, textpos)

        pygame.display.flip()


        # Wait
        pygame.time.wait(int(wait_time * 1000))

if __name__ == "__main__":
    main()

