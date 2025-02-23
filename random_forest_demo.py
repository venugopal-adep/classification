import pygame
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import io

# Initialize Pygame
pygame.init()

# Set up the display with 1920x1080 resolution
WIDTH, HEIGHT = 1920, 1080
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Random Forest Interactive Visualization")

# Colors with alpha support
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)
LIGHT_GRAY = (240, 240, 240)
TRANSPARENT_RED = (255, 0, 0, 50)
TRANSPARENT_BLUE = (0, 0, 255, 50)

# Enhanced fonts
title_font = pygame.font.Font(None, 72)
subtitle_font = pygame.font.Font(None, 36)
text_font = pygame.font.Font(None, 28)
formula_font = pygame.font.Font(None, 32)

# Data structures
points = []
labels = []
model = None
model_trained = False
metrics = None
forest_image = None

# Button dimensions and positions
BUTTON_WIDTH = 120
BUTTON_HEIGHT = 50
BUTTON_MARGIN = 20

# Define buttons
clear_button = pygame.Rect(50, HEIGHT - 80, BUTTON_WIDTH, BUTTON_HEIGHT)
train_button = pygame.Rect(190, HEIGHT - 80, BUTTON_WIDTH, BUTTON_HEIGHT)
view_forest_button = pygame.Rect(330, HEIGHT - 80, 160, BUTTON_HEIGHT)

# Define visualization areas
plot_area = pygame.Rect(50, 120, WIDTH//2 - 100, HEIGHT - 250)
metrics_area = pygame.Rect(WIDTH//2 + 50, HEIGHT//2 - 200, WIDTH//2 - 100, HEIGHT//2 + 100)

def add_point(pos, label):
    if plot_area.collidepoint(pos):
        global model_trained
        points.append(pos)
        labels.append(label)
        model_trained = False

def clear_data():
    global points, labels, model_trained, model, metrics, forest_image
    points = []
    labels = []
    model_trained = False
    model = None
    metrics = None
    forest_image = None

def train_model():
    global model, model_trained, metrics
    if len(points) > 1 and len(set(labels)) > 1:
        X = np.array(points)
        y = np.array(labels)
        model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
        model.fit(X, y)
        model_trained = True
        
        # Calculate metrics
        y_pred = model.predict(X)
        metrics = {
            'Accuracy': accuracy_score(y, y_pred),
            'Precision': precision_score(y, y_pred, average='binary'),
            'Recall': recall_score(y, y_pred, average='binary'),
            'F1-score': f1_score(y, y_pred, average='binary')
        }

def show_forest_structure():
    if model is not None:
        forest_window = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Random Forest Structure Visualization")
        
        plt.style.use('dark_background')
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        axes = axes.ravel()
        
        for i, tree in enumerate(model.estimators_[:12]):
            plot_tree(tree, ax=axes[i], feature_names=['X', 'Y'], 
                     filled=True, rounded=True, fontsize=8)
            axes[i].set_title(f"Decision Tree {i+1}")
        
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
        buf.seek(0)
        forest_surface = pygame.image.load(buf)
        forest_surface = pygame.transform.scale(forest_surface, (WIDTH, HEIGHT))
        plt.close(fig)
        
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (
                    event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False
            
            forest_window.fill(WHITE)
            forest_window.blit(forest_surface, (0, 0))
            
            # Add instructions
            instruction = text_font.render("Press ESC to return to main window", True, BLACK)
            forest_window.blit(instruction, (WIDTH//2 - instruction.get_width()//2, HEIGHT - 40))
            
            pygame.display.flip()
        
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Random Forest Interactive Visualization")

def draw_decision_boundary():
    if model_trained and model is not None:
        xx, yy = np.meshgrid(np.arange(plot_area.left, plot_area.right, 5),
                            np.arange(plot_area.top, plot_area.bottom, 5))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        surface = pygame.Surface((plot_area.width, plot_area.height), pygame.SRCALPHA)
        
        for i in range(len(xx)):
            for j in range(len(yy)):
                color = TRANSPARENT_RED if Z[i][j] == 0 else TRANSPARENT_BLUE
                pygame.draw.circle(surface, color, 
                                 (int(xx[i][j] - plot_area.left), 
                                  int(yy[i][j] - plot_area.top)), 3)
        
        screen.blit(surface, plot_area)

def draw_metrics():
    if metrics is not None:
        pygame.draw.rect(screen, LIGHT_GRAY, metrics_area)
        metrics_title = subtitle_font.render("Model Metrics with Formulas", True, BLACK)
        screen.blit(metrics_title, (metrics_area.centerx - metrics_title.get_width()//2, 
                                  metrics_area.top - 30))
        
        formulas = {
            'Accuracy': "TP + TN / (TP + TN + FP + FN)",
            'Precision': "TP / (TP + FP)",
            'Recall': "TP / (TP + FN)",
            'F1-score': "2 × (Precision × Recall) / (Precision + Recall)"
        }
        
        for i, (metric, value) in enumerate(metrics.items()):
            # Metric name
            metric_text = formula_font.render(f"{metric}:", True, BLACK)
            screen.blit(metric_text, (metrics_area.left + 20, 
                                    metrics_area.top + 20 + i * 80))
            
            # Formula
            formula_text = text_font.render(f"Formula: {formulas[metric]}", True, BLACK)
            screen.blit(formula_text, (metrics_area.left + 40, 
                                     metrics_area.top + 45 + i * 80))
            
            # Value
            value_text = text_font.render(f"Value: {value:.4f}", True, BLUE)
            screen.blit(value_text, (metrics_area.left + 40, 
                                   metrics_area.top + 65 + i * 80))

def draw():
    screen.fill(WHITE)
    
    # Draw title and subtitle
    title = title_font.render("Random Forest Interactive Visualization", True, BLACK)
    subtitle = subtitle_font.render("Left Click: Class 0 (Red) | Right Click: Class 1 (Blue)", True, BLACK)
    
    screen.blit(title, (WIDTH//2 - title.get_width()//2, 20))
    screen.blit(subtitle, (WIDTH//2 - subtitle.get_width()//2, 70))
    
    # Draw plot area
    pygame.draw.rect(screen, LIGHT_GRAY, plot_area)
    if model_trained:
        draw_decision_boundary()
    
    # Draw points
    for point, label in zip(points, labels):
        color = RED if label == 0 else BLUE
        pygame.draw.circle(screen, color, point, 6)
    
    # Draw metrics
    draw_metrics()
    
    # Draw buttons
    pygame.draw.rect(screen, YELLOW, clear_button)
    pygame.draw.rect(screen, GREEN, train_button)
    pygame.draw.rect(screen, BLUE, view_forest_button)
    
    clear_text = text_font.render("Clear", True, BLACK)
    train_text = text_font.render("Train", True, BLACK)
    forest_text = text_font.render("View Forest", True, WHITE)
    
    screen.blit(clear_text, (clear_button.centerx - clear_text.get_width()//2,
                            clear_button.centery - clear_text.get_height()//2))
    screen.blit(train_text, (train_button.centerx - train_text.get_width()//2,
                            train_button.centery - train_text.get_height()//2))
    screen.blit(forest_text, (view_forest_button.centerx - forest_text.get_width()//2,
                             view_forest_button.centery - forest_text.get_height()//2))
    
    pygame.display.flip()

# Main loop
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
            elif view_forest_button.collidepoint(event.pos):
                show_forest_structure()
            elif plot_area.collidepoint(event.pos):
                if event.button == 1:  # Left click
                    add_point(event.pos, 0)
                elif event.button == 3:  # Right click
                    add_point(event.pos, 1)
    
    draw()

pygame.quit()
