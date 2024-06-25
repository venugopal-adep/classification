import pygame
import random
import math
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Initialize Pygame
pygame.init()

# Set up the display
WIDTH, HEIGHT = 1600, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Decision Tree vs Random Forest Demo")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GRAY = (200, 200, 200)

# Fonts
title_font = pygame.font.Font(None, 48)
text_font = pygame.font.Font(None, 24)
button_font = pygame.font.Font(None, 32)

# Classes for demo
class Point:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color

class Button:
    def __init__(self, x, y, width, height, text, color, text_color, action):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.text_color = text_color
        self.action = action

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect)
        text_surf = button_font.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        screen.blit(text_surf, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.action()

# Helper functions
def generate_circular_data(num_points, noise=0.1):
    points = []
    for _ in range(num_points):
        angle = random.uniform(0, 2 * math.pi)
        radius = random.uniform(0, 1)
        x = radius * math.cos(angle) + random.gauss(0, noise)
        y = radius * math.sin(angle) + random.gauss(0, noise)
        color = RED if radius <= 0.5 else BLUE
        points.append(Point(x, y, color))
    return points

def scale_point(point, scale_x, scale_y, offset_x, offset_y):
    return Point(point.x * scale_x + offset_x, point.y * scale_y + offset_y, point.color)

def draw_decision_boundary(classifier, screen, offset_x, offset_y, scale_x, scale_y):
    xx, yy = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    for i in range(Z.shape[0] - 1):
        for j in range(Z.shape[1] - 1):
            if Z[i, j] != Z[i+1, j] or Z[i, j] != Z[i, j+1]:
                x1, y1 = xx[i, j] * scale_x + offset_x, yy[i, j] * scale_y + offset_y
                x2, y2 = xx[i+1, j] * scale_x + offset_x, yy[i+1, j] * scale_y + offset_y
                pygame.draw.line(screen, BLACK, (x1, y1), (x2, y2), 1)
                x2, y2 = xx[i, j+1] * scale_x + offset_x, yy[i, j+1] * scale_y + offset_y
                pygame.draw.line(screen, BLACK, (x1, y1), (x2, y2), 1)

# Create objects
points = generate_circular_data(200)
dt_classifier = DecisionTreeClassifier(max_depth=5)
rf_classifier = RandomForestClassifier(n_estimators=10, max_depth=5)

# Train classifiers
X = [(p.x, p.y) for p in points]
y = [0 if p.color == RED else 1 for p in points]
dt_classifier.fit(X, y)
rf_classifier.fit(X, y)

# Create buttons
def clear_points():
    global points, dt_classifier, rf_classifier
    points = generate_circular_data(200)
    X = [(p.x, p.y) for p in points]
    y = [0 if p.color == RED else 1 for p in points]
    dt_classifier.fit(X, y)
    rf_classifier.fit(X, y)

clear_button = Button(WIDTH // 2 - 100, HEIGHT - 60, 200, 50, "Generate New Data", GRAY, BLACK, clear_points)

# Main game loop
running = True
clock = pygame.time.Clock()
show_accuracy = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            clear_button.handle_event(event)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                show_accuracy = not show_accuracy

    # Clear the screen
    screen.fill(WHITE)

    # Draw title and developer info
    title_text = title_font.render("Decision Tree vs Random Forest Demo", True, BLACK)
    screen.blit(title_text, (WIDTH // 2 - title_text.get_width() // 2, 20))
    
    dev_text = text_font.render("Developed by: Venugopal Adep", True, BLACK)
    screen.blit(dev_text, (WIDTH // 2 - dev_text.get_width() // 2, 70))

    # Draw decision boundaries
    draw_decision_boundary(dt_classifier, screen, WIDTH // 4, HEIGHT // 2, WIDTH // 4, HEIGHT // 2)
    draw_decision_boundary(rf_classifier, screen, WIDTH * 3 // 4, HEIGHT // 2, WIDTH // 4, HEIGHT // 2)

    # Draw points
    for point in points:
        dt_point = scale_point(point, WIDTH // 4, HEIGHT // 2, WIDTH // 4, HEIGHT // 2)
        rf_point = scale_point(point, WIDTH // 4, HEIGHT // 2, WIDTH * 3 // 4, HEIGHT // 2)
        pygame.draw.circle(screen, point.color, (int(dt_point.x), int(dt_point.y)), 3)
        pygame.draw.circle(screen, point.color, (int(rf_point.x), int(rf_point.y)), 3)

    # Draw labels
    dt_text = text_font.render("Decision Tree", True, BLACK)
    screen.blit(dt_text, (WIDTH // 4 - dt_text.get_width() // 2, 150))
    rf_text = text_font.render("Random Forest", True, BLACK)
    screen.blit(rf_text, (WIDTH * 3 // 4 - rf_text.get_width() // 2, 150))

    # Draw button
    clear_button.draw(screen)

    # Show accuracy
    if show_accuracy:
        dt_accuracy = dt_classifier.score(X, y) * 100
        rf_accuracy = rf_classifier.score(X, y) * 100
        
        dt_acc_text = text_font.render(f"Decision Tree Accuracy: {dt_accuracy:.2f}%", True, BLACK)
        rf_acc_text = text_font.render(f"Random Forest Accuracy: {rf_accuracy:.2f}%", True, BLACK)
        
        screen.blit(dt_acc_text, (50, HEIGHT - 100))
        screen.blit(rf_acc_text, (WIDTH // 2 + 50, HEIGHT - 100))

    # Draw instructions
    instructions = [
        "Press SPACE to toggle accuracy display",
        "Click 'Generate New Data' to create a new dataset"
    ]
    for i, instruction in enumerate(instructions):
        inst_text = text_font.render(instruction, True, BLACK)
        screen.blit(inst_text, (WIDTH // 2 - inst_text.get_width() // 2, 100 + i * 30))

    # Update the display
    pygame.display.flip()
    clock.tick(60)

# Quit Pygame
pygame.quit()