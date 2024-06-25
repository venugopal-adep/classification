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
pygame.display.set_caption("Decision Tree vs Random Forest Demo: Fruit Classification")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (200, 200, 200)

# Fonts
title_font = pygame.font.Font(None, 48)
text_font = pygame.font.Font(None, 24)
button_font = pygame.font.Font(None, 32)

# Fruit class
class Fruit:
    def __init__(self, weight, sweetness, type):
        self.weight = weight
        self.sweetness = sweetness
        self.type = type

# Button class
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
def generate_fruit_data(num_fruits):
    fruits = []
    for _ in range(num_fruits):
        weight = random.uniform(0, 1)
        sweetness = random.uniform(0, 1)
        
        if weight < 0.4 and sweetness > 0.6:
            fruit_type = "strawberry"
        elif weight > 0.6 and sweetness < 0.4:
            fruit_type = "apple"
        elif weight > 0.5 and sweetness > 0.5:
            fruit_type = "orange"
        else:
            fruit_type = random.choice(["strawberry", "apple", "orange"])
        
        fruits.append(Fruit(weight, sweetness, fruit_type))
    return fruits

def draw_fruit(screen, fruit, x, y):
    if fruit.type == "strawberry":
        color = RED
    elif fruit.type == "apple":
        color = GREEN
    else:  # orange
        color = YELLOW
    pygame.draw.circle(screen, color, (int(x), int(y)), 5)

def draw_decision_boundary(classifier, screen, offset_x, offset_y, scale_x, scale_y):
    xx, yy = np.meshgrid(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
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
fruits = generate_fruit_data(200)
dt_classifier = DecisionTreeClassifier(max_depth=5)
rf_classifier = RandomForestClassifier(n_estimators=10, max_depth=5)

# Train classifiers
X = [(f.weight, f.sweetness) for f in fruits]
y = [f.type for f in fruits]
dt_classifier.fit(X, y)
rf_classifier.fit(X, y)

# Create button
def generate_new_data():
    global fruits, dt_classifier, rf_classifier
    fruits = generate_fruit_data(200)
    X = [(f.weight, f.sweetness) for f in fruits]
    y = [f.type for f in fruits]
    dt_classifier.fit(X, y)
    rf_classifier.fit(X, y)

generate_button = Button(WIDTH // 2 - 100, HEIGHT - 60, 200, 50, "Generate New Data", GRAY, BLACK, generate_new_data)

# Main game loop
running = True
clock = pygame.time.Clock()
show_accuracy = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            generate_button.handle_event(event)
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                show_accuracy = not show_accuracy

    # Clear the screen
    screen.fill(WHITE)

    # Draw title and developer info
    title_text = title_font.render("Decision Tree vs Random Forest: Fruit Classification", True, BLACK)
    screen.blit(title_text, (WIDTH // 2 - title_text.get_width() // 2, 20))
    
    dev_text = text_font.render("Developed by: Venugopal Adep", True, BLACK)
    screen.blit(dev_text, (WIDTH // 2 - dev_text.get_width() // 2, 70))

    # Draw decision boundaries
    draw_decision_boundary(dt_classifier, screen, WIDTH // 4, HEIGHT // 4, WIDTH // 4, HEIGHT // 2)
    draw_decision_boundary(rf_classifier, screen, WIDTH * 3 // 4, HEIGHT // 4, WIDTH // 4, HEIGHT // 2)

    # Draw fruits
    for fruit in fruits:
        dt_x = fruit.weight * WIDTH // 4 + WIDTH // 4
        dt_y = (1 - fruit.sweetness) * HEIGHT // 2 + HEIGHT // 4
        rf_x = fruit.weight * WIDTH // 4 + WIDTH * 3 // 4
        rf_y = (1 - fruit.sweetness) * HEIGHT // 2 + HEIGHT // 4
        draw_fruit(screen, fruit, dt_x, dt_y)
        draw_fruit(screen, fruit, rf_x, rf_y)

    # Draw labels
    dt_text = text_font.render("Decision Tree", True, BLACK)
    screen.blit(dt_text, (WIDTH // 4 - dt_text.get_width() // 2, HEIGHT // 4 - 30))
    rf_text = text_font.render("Random Forest", True, BLACK)
    screen.blit(rf_text, (WIDTH * 3 // 4 - rf_text.get_width() // 2, HEIGHT // 4 - 30))

    # Draw axes labels
    weight_text = text_font.render("Weight →", True, BLACK)
    sweetness_text = text_font.render("Sweetness →", True, BLACK)
    screen.blit(weight_text, (WIDTH // 4 - weight_text.get_width() // 2, HEIGHT * 3 // 4 + 10))
    screen.blit(weight_text, (WIDTH * 3 // 4 - weight_text.get_width() // 2, HEIGHT * 3 // 4 + 10))
    sweetness_text = pygame.transform.rotate(sweetness_text, 90)
    screen.blit(sweetness_text, (WIDTH // 4 - 50, HEIGHT // 2 - sweetness_text.get_width() // 2))
    screen.blit(sweetness_text, (WIDTH * 3 // 4 - 50, HEIGHT // 2 - sweetness_text.get_width() // 2))

    # Draw legend
    legend_items = [
        ("Strawberry", RED),
        ("Apple", GREEN),
        ("Orange", YELLOW)
    ]
    for i, (fruit_name, color) in enumerate(legend_items):
        pygame.draw.circle(screen, color, (WIDTH // 2 - 100, HEIGHT - 120 + i * 30), 5)
        legend_text = text_font.render(fruit_name, True, BLACK)
        screen.blit(legend_text, (WIDTH // 2 - 80, HEIGHT - 125 + i * 30))

    # Draw button
    generate_button.draw(screen)

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