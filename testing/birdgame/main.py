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

    bird_colors = {
        0: (0, 0, 0)
    }
    
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
                if event.key == pygame.K_r:
                    game.reset()
            

        if jump:
            if not game.birds[0].dead:
                game.birds[0].jump()
        # Game logic
        game.tick(wait_time)

        score = int(game.score*10)

        current_zero = game.score * SCREEN_HEIGHT - bird_pos

        # Draw
        screen.fill(SKY_BLUE)
        for pipe in game.pipes:
            pipe_x = pipe.x * SCREEN_HEIGHT - current_zero
            if pipe_x > SCREEN_WIDTH or pipe_x + pipe.width * SCREEN_HEIGHT < 0:
                # dont draw pipes that are off screen
                continue
            pipe_w = pipe.width * SCREEN_HEIGHT
            gap_b = pipe.gap_y * SCREEN_HEIGHT
            gap_t = SCREEN_HEIGHT - (pipe.gap_y + pipe.gap_size) * SCREEN_HEIGHT
            pygame.draw.rect(screen, GREEN, pygame.Rect(pipe_x, 0, pipe_w, gap_t))
            pygame.draw.rect(screen, GREEN, pygame.Rect(pipe_x, SCREEN_HEIGHT - gap_b, pipe_w, gap_b))
        for i, bird in enumerate(game.birds):
            bird_c = BLACK
            if i not in bird_colors:
                bird_colors[i] = generate_bird_color()
            bird_c = bird_colors[i]

            bpos = bird_pos
            if bird.dead:
                bpos = bird_pos - (game.score - bird.score) * SCREEN_HEIGHT
            
            pygame.draw.rect(screen, bird_c, pygame.Rect(bpos, (1 - bird.y-bird.size) * SCREEN_HEIGHT, bird.size * SCREEN_HEIGHT, bird.size * SCREEN_HEIGHT))


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

