import pygame
import math
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Initialize Pygame
pygame.init()

# Set up the display
WIDTH, HEIGHT = 1600, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Logistic Regression Demo")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
LIGHT_GRAY = (200, 200, 200)

# Fonts
title_font = pygame.font.Font(None, 64)
subtitle_font = pygame.font.Font(None, 32)
text_font = pygame.font.Font(None, 24)

# Data
points = []
labels = []

# Logistic Regression model
model = None
model_trained = False
metrics = None

# Buttons
clear_button = pygame.Rect(50, HEIGHT - 60, 100, 40)
train_button = pygame.Rect(170, HEIGHT - 60, 100, 40)

def add_point(pos, label):
    global model_trained
    points.append(pos)
    labels.append(label)
    model_trained = False

def clear_data():
    global points, labels, model_trained, model, metrics
    points = []
    labels = []
    model_trained = False
    model = None
    metrics = None

def train_model():
    global model, model_trained, metrics
    if len(points) > 1 and len(set(labels)) > 1:
        X = np.array(points)
        y = np.array(labels)
        model = LogisticRegression()
        model.fit(X, y)
        model_trained = True
        y_pred = model.predict(X)
        metrics = {
            'Accuracy': accuracy_score(y, y_pred),
            'Precision': precision_score(y, y_pred, average='binary'),
            'Recall': recall_score(y, y_pred, average='binary'),
            'F1-score': f1_score(y, y_pred, average='binary')
        }

def draw_decision_boundary():
    if model_trained and model is not None:
        xx, yy = np.meshgrid(np.arange(0, WIDTH//2, 5), np.arange(0, HEIGHT-100, 5))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        for i in range(len(xx)):
            for j in range(len(yy)):
                if Z[i][j] == 0:
                    pygame.draw.circle(screen, (*RED, 50), (int(xx[i][j]), int(yy[i][j])), 2)
                else:
                    pygame.draw.circle(screen, (*BLUE, 50), (int(xx[i][j]), int(yy[i][j])), 2)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def draw_sigmoid():
    sigmoid_rect = pygame.Rect(WIDTH//2 + 50, 120, WIDTH//2 - 100, HEIGHT//2 - 140)
    pygame.draw.rect(screen, LIGHT_GRAY, sigmoid_rect)
    
    pygame.draw.line(screen, BLACK, (sigmoid_rect.left, sigmoid_rect.bottom), (sigmoid_rect.right, sigmoid_rect.bottom), 2)
    pygame.draw.line(screen, BLACK, (sigmoid_rect.left, sigmoid_rect.top), (sigmoid_rect.left, sigmoid_rect.bottom), 2)
    
    for x in range(sigmoid_rect.width):
        y1 = sigmoid((x - sigmoid_rect.width//2) / (sigmoid_rect.width//10)) * sigmoid_rect.height
        y2 = sigmoid((x + 1 - sigmoid_rect.width//2) / (sigmoid_rect.width//10)) * sigmoid_rect.height
        pygame.draw.line(screen, GREEN, 
                         (sigmoid_rect.left + x, sigmoid_rect.bottom - y1), 
                         (sigmoid_rect.left + x + 1, sigmoid_rect.bottom - y2), 2)

def draw_metrics():
    if metrics is not None:
        metrics_rect = pygame.Rect(WIDTH//2 + 50, HEIGHT//2 + 20, WIDTH//2 - 100, HEIGHT//2 - 140)
        pygame.draw.rect(screen, LIGHT_GRAY, metrics_rect)
        
        for i, (metric, value) in enumerate(metrics.items()):
            text = subtitle_font.render(f"{metric}: {value:.4f}", True, BLACK)
            screen.blit(text, (metrics_rect.left + 20, metrics_rect.top + 20 + i * 40))

def draw():
    screen.fill(WHITE)
    
    # Draw title and subtitle
    title = title_font.render("Logistic Regression Demo", True, BLACK)
    screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 20))
    
    subtitle = subtitle_font.render("Developed by: Venugopal Adep", True, BLACK)
    screen.blit(subtitle, (WIDTH // 2 - subtitle.get_width() // 2, 80))
    
    # Draw decision boundary and points
    pygame.draw.rect(screen, LIGHT_GRAY, (0, 100, WIDTH//2, HEIGHT-160))
    draw_decision_boundary()
    for point, label in zip(points, labels):
        color = RED if label == 0 else BLUE
        pygame.draw.circle(screen, color, point, 5)
    
    # Draw sigmoid function
    draw_sigmoid()
    sigmoid_title = subtitle_font.render("Sigmoid Function", True, BLACK)
    screen.blit(sigmoid_title, (WIDTH * 3 // 4 - sigmoid_title.get_width() // 2, 120))
    
    # Draw metrics
    draw_metrics()
    if metrics is not None:
        metrics_title = subtitle_font.render("Model Metrics", True, BLACK)
        screen.blit(metrics_title, (WIDTH * 3 // 4 - metrics_title.get_width() // 2, HEIGHT // 2 + 20))
    
    # Draw buttons
    pygame.draw.rect(screen, YELLOW, clear_button)
    pygame.draw.rect(screen, GREEN, train_button)
    
    clear_text = text_font.render("Clear", True, BLACK)
    train_text = text_font.render("Train", True, BLACK)
    
    screen.blit(clear_text, (clear_button.centerx - clear_text.get_width() // 2, 
                             clear_button.centery - clear_text.get_height() // 2))
    screen.blit(train_text, (train_button.centerx - train_text.get_width() // 2, 
                             train_button.centery - train_text.get_height() // 2))
    
    # Draw instructions
    instructions = [
        "Left-click: Add red point (Class 0)",
        "Right-click: Add blue point (Class 1)",
        "Train: Fit logistic regression model",
        "Clear: Remove all points",
    ]
    
    for i, instruction in enumerate(instructions):
        text = text_font.render(instruction, True, BLACK)
        screen.blit(text, (10, HEIGHT - 150 + i * 30))
    
    pygame.display.flip()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if clear_button.collidepoint(event.pos):
                clear_data()
            elif train_button.collidepoint(event.pos):
                train_model()
            elif event.pos[0] < WIDTH//2 and event.pos[1] < HEIGHT-160:
                if event.button == 1:  # Left click
                    add_point(event.pos, 0)
                elif event.button == 3:  # Right click
                    add_point(event.pos, 1)
    
    draw()

pygame.quit()