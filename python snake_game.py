# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 22:43:09 2023

@author: rakes
"""

import pygame
import sys
import random

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 400
GRID_SIZE = 15  # Adjust the grid size
GRID_WIDTH = SCREEN_WIDTH // GRID_SIZE
GRID_HEIGHT = SCREEN_HEIGHT // GRID_SIZE
SNAKE_SPEED = 10  # Initial speed
MAX_SPEED = 20  # Maximum speed
SPEED_INCREMENT = 1  # Speed increment per score point

# Colors
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# Initialize the screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption('Snake Game')

# Initialize the snake
snake = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
snake_direction = (0, 0)  # Initialize with no movement
snake_length = 1

# Initialize the food
food = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))

# Initialize score
score = 0

# Variable to track if the game is started
game_started = False
game_over = False

# Main game loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.KEYDOWN:
            if not game_started:
                # Start a new game when the first arrow key is pressed
                snake = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
                snake_direction = (0, 0)
                snake_length = 1
                score = 0  # Reset the score to zero
                game_started = True
                snake_direction = (0, 1)  # Start moving down
            if event.key == pygame.K_UP and snake_direction != (0, 1):
                snake_direction = (0, -1)
            elif event.key == pygame.K_DOWN and snake_direction != (0, -1):
                snake_direction = (0, 1)
            elif event.key == pygame.K_LEFT and snake_direction != (1, 0):
                snake_direction = (-1, 0)
            elif event.key == pygame.K_RIGHT and snake_direction != (-1, 0):
                snake_direction = (1, 0)

    if game_started:
        # Rest of the game logic remains the same

        # Move the snake
        x, y = snake[0]
        new_head = ((x + snake_direction[0]) % GRID_WIDTH, (y + snake_direction[1]) % GRID_HEIGHT)  # Wrap around
        snake.insert(0, new_head)

        # Check for collisions
        if snake[0] == food:
            food = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
            snake_length += 1
            score += 1  # Increase the score when food is eaten

            # Increase speed when score increases
            if SNAKE_SPEED < MAX_SPEED:
                SNAKE_SPEED += SPEED_INCREMENT

        if len(snake) > snake_length:
            snake.pop()

        if new_head in snake[1:]:
            game_over = True
            game_started = False

        else:
            screen.fill(BLACK)
            pygame.draw.rect(screen, RED, (food[0] * GRID_SIZE, food[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))
            for segment in snake:
                pygame.draw.rect(screen, GREEN, (segment[0] * GRID_SIZE, segment[1] * GRID_SIZE, GRID_SIZE, GRID_SIZE))

        # Display the score in the top-middle of the screen
        font = pygame.font.Font(None, 36)
        score_display = font.render(f"Score: {score}", True, GREEN)
        score_rect = score_display.get_rect(center=(SCREEN_WIDTH // 2, 20))
        screen.blit(score_display, score_rect)
    else:
        if game_over:
            # Show "Game Over" in the middle of the screen
            font = pygame.font.Font(None, 36)
            text = font.render("Game Over", True, GREEN)
            text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 20))
            screen.blit(text, text_rect)

            # Display the final score where the continually updated score was
            score_message = f"Score: {score}"
            score_text = font.render(score_message, True, GREEN)
            score_rect = score_text.get_rect(center=(SCREEN_WIDTH // 2, 20))
            screen.blit(score_text, score_rect)

            # Display "Press an arrow key to start" below "Game Over"
            restart_text = font.render("Press an arrow key to start", True, GREEN)
            restart_rect = restart_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 20))
            screen.blit(restart_text, restart_rect)
        else:
            # Show a message to start the game
            font = pygame.font.Font(None, 36)
            text = font.render("Press an arrow key to start", True, GREEN)
            text_rect = text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 40))

            # Display the score in the top-middle of the screen
            score_message = f"Score: {score}"
            score_text = font.render(score_message, True, GREEN)
            score_rect = score_text.get_rect(center=(SCREEN_WIDTH // 2, 20))
            screen.blit(text, text_rect)
            screen.blit(score_text, score_rect)

    pygame.display.update()
    pygame.time.Clock().tick(SNAKE_SPEED)

