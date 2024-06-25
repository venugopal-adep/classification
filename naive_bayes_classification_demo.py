import pygame
import sys
import random
import math

# Initialize Pygame
pygame.init()

# Set up the display
WIDTH, HEIGHT = 1600, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Naive Bayes Classification")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
LIGHT_RED = (255, 200, 200)
LIGHT_BLUE = (200, 200, 255)

# Fonts
title_font = pygame.font.Font(None, 64)
text_font = pygame.font.Font(None, 32)
small_font = pygame.font.Font(None, 24)

# Data points
class_a = []
class_b = []

# Classification result
result = None

# Visualization settings
point_radius = 5
grid_size = 20

def generate_data():
    global class_a, class_b
    class_a = [(random.gauss(WIDTH/4, WIDTH/16), random.gauss(HEIGHT/2, HEIGHT/8)) for _ in range(50)]
    class_b = [(random.gauss(3*WIDTH/4, WIDTH/16), random.gauss(HEIGHT/2, HEIGHT/8)) for _ in range(50)]

def draw_text(text, font, color, x, y):
    surface = font.render(text, True, color)
    screen.blit(surface, (x, y))

def gaussian_probability(x, mean, std_dev):
    return math.exp(-((x - mean) ** 2) / (2 * std_dev ** 2)) / (std_dev * math.sqrt(2 * math.pi))

def naive_bayes_classify(point):
    prob_a = len(class_a) / (len(class_a) + len(class_b))
    prob_b = len(class_b) / (len(class_a) + len(class_b))
    
    mean_a_x = sum(p[0] for p in class_a) / len(class_a)
    mean_a_y = sum(p[1] for p in class_a) / len(class_a)
    std_dev_a_x = math.sqrt(sum((p[0] - mean_a_x) ** 2 for p in class_a) / len(class_a))
    std_dev_a_y = math.sqrt(sum((p[1] - mean_a_y) ** 2 for p in class_a) / len(class_a))
    
    mean_b_x = sum(p[0] for p in class_b) / len(class_b)
    mean_b_y = sum(p[1] for p in class_b) / len(class_b)
    std_dev_b_x = math.sqrt(sum((p[0] - mean_b_x) ** 2 for p in class_b) / len(class_b))
    std_dev_b_y = math.sqrt(sum((p[1] - mean_b_y) ** 2 for p in class_b) / len(class_b))
    
    prob_a *= gaussian_probability(point[0], mean_a_x, std_dev_a_x) * gaussian_probability(point[1], mean_a_y, std_dev_a_y)
    prob_b *= gaussian_probability(point[0], mean_b_x, std_dev_b_x) * gaussian_probability(point[1], mean_b_y, std_dev_b_y)
    
    return "Class A" if prob_a > prob_b else "Class B"

def draw_grid():
    for x in range(0, WIDTH, grid_size):
        for y in range(0, HEIGHT, grid_size):
            classification = naive_bayes_classify((x, y))
            color = LIGHT_RED if classification == "Class A" else LIGHT_BLUE
            pygame.draw.rect(screen, color, (x, y, grid_size, grid_size))

# Generate initial data
generate_data()

# Main game loop
clock = pygame.time.Clock()
show_grid = False

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left mouse button
                x, y = event.pos
                result = naive_bayes_classify((x, y))
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                generate_data()
                result = None
            elif event.key == pygame.K_g:
                show_grid = not show_grid

    screen.fill(WHITE)

    # Draw classification grid
    if show_grid:
        draw_grid()

    # Draw title and developer info
    draw_text("Naive Bayes Classification", title_font, BLACK, 20, 20)
    draw_text("Developed by: Venugopal Adep", text_font, BLACK, 20, 80)

    # Draw instructions
    draw_text("Left-click to classify a point", text_font, BLACK, 20, HEIGHT - 90)
    draw_text("Press R to regenerate data", text_font, BLACK, 20, HEIGHT - 60)
    draw_text("Press G to toggle classification grid", text_font, BLACK, 20, HEIGHT - 30)

    # Draw data points
    for point in class_a:
        pygame.draw.circle(screen, RED, (int(point[0]), int(point[1])), point_radius)
    for point in class_b:
        pygame.draw.circle(screen, BLUE, (int(point[0]), int(point[1])), point_radius)

    # Draw classification result
    if result:
        x, y = pygame.mouse.get_pos()
        color = RED if result == "Class A" else BLUE
        pygame.draw.circle(screen, color, (x, y), point_radius * 2, 2)
        draw_text(f"Classification: {result}", text_font, BLACK, x + 20, y - 20)

    # Draw legend
    draw_text("Class A", small_font, RED, WIDTH - 100, 20)
    draw_text("Class B", small_font, BLUE, WIDTH - 100, 50)

    pygame.display.flip()
    clock.tick(60)