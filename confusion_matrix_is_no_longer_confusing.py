import pygame
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Initialize Pygame
pygame.init()

# Set up the display
WIDTH, HEIGHT = 1800, 900
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Confusion Matrix Explorer")

# Colors
BACKGROUND = (240, 240, 245)
LIGHT_RED = (255, 200, 200)
LIGHT_BLUE = (200, 200, 255)
TEXT_COLOR = (50, 50, 50)
POSITIVE_COLOR = (255, 100, 100)  # Red for positive class
NEGATIVE_COLOR = (100, 100, 255)  # Blue for negative class
GREEN = (100, 200, 100)
YELLOW = (255, 255, 100)
LIGHT_GRAY = (220, 220, 220)
PURPLE = (200, 100, 200)
CYAN = (100, 200, 200)

# Fonts
title_font = pygame.font.Font(None, 64)
subtitle_font = pygame.font.Font(None, 36)
text_font = pygame.font.Font(None, 24)

# Data
points = []
labels = []

# Logistic Regression model
model = None
model_trained = False
metrics = None

# Buttons
clear_button = pygame.Rect(50, HEIGHT - 70, 180, 50)
train_button = pygame.Rect(250, HEIGHT - 70, 180, 50)

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

def calculate_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1,
        'TP(RpR)': tp, 'TN(BpB)': tn, 'FP(BpR)': fp, 'FN(RpB)': fn
    }

def train_model():
    global model, model_trained, metrics
    if len(points) > 1 and len(set(labels)) > 1:
        X = np.array(points)
        y = np.array(labels)
        model = LogisticRegression()
        model.fit(X, y)
        model_trained = True
        y_pred = model.predict(X)
        metrics = calculate_metrics(y, y_pred)

def draw_decision_boundary():
    if model_trained and model is not None:
        xx, yy = np.meshgrid(np.arange(0, WIDTH//2, 5), np.arange(100, HEIGHT-100, 5))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        
        for i in range(len(xx)):
            for j in range(len(yy)):
                if Z[i][j] == 1:  # Positive class (light red)
                    pygame.draw.circle(screen, (*LIGHT_RED, 100), (int(xx[i][j]), int(yy[i][j])), 3)
                else:  # Negative class (light blue)
                    pygame.draw.circle(screen, (*LIGHT_BLUE, 100), (int(xx[i][j]), int(yy[i][j])), 3)

def draw_metrics():
    metrics_rect = pygame.Rect(WIDTH//2 + 10, 120, WIDTH//2 - 20, HEIGHT - 200)
    pygame.draw.rect(screen, LIGHT_GRAY, metrics_rect)
    
    if metrics is not None:
        y_offset = 10
        colors = [CYAN, GREEN, YELLOW, PURPLE]
        for i, (metric, value) in enumerate(metrics.items()):
            if metric in ['TP(RpR)', 'TN(BpB)', 'FP(BpR)', 'FN(RpB)']:
                continue
            pygame.draw.rect(screen, colors[i], (metrics_rect.left + 10, metrics_rect.top + y_offset, metrics_rect.width - 20, 40))
            text = subtitle_font.render(f"{metric}: {value:.4f}", True, TEXT_COLOR)
            screen.blit(text, (metrics_rect.left + 20, metrics_rect.top + y_offset + 10))
            y_offset += 50
        
        # Confusion matrix
        cm_rect = pygame.Rect(metrics_rect.left + 10, metrics_rect.top + y_offset, metrics_rect.width - 20, 200)
        pygame.draw.rect(screen, (200, 200, 200), cm_rect)
        pygame.draw.line(screen, TEXT_COLOR, (cm_rect.centerx, cm_rect.top), (cm_rect.centerx, cm_rect.bottom), 2)
        pygame.draw.line(screen, TEXT_COLOR, (cm_rect.left, cm_rect.centery), (cm_rect.right, cm_rect.centery), 2)
        
        cm_title = text_font.render("Actual Values", True, TEXT_COLOR)
        screen.blit(cm_title, (cm_rect.centerx - cm_title.get_width()//2, cm_rect.top - 30))
        
        pos_neg_labels = ["Positive (1)", "Negative (0)"]
        for i, label in enumerate(pos_neg_labels):
            text = text_font.render(label, True, TEXT_COLOR)
            screen.blit(text, (cm_rect.left + i * (cm_rect.width//2) + cm_rect.width//4 - text.get_width()//2, cm_rect.top - 10))
            screen.blit(text, (cm_rect.left - text.get_width() - 10, cm_rect.top + i * (cm_rect.height//2) + cm_rect.height//4 - text.get_height()//2))
        
        pred_values_text = text_font.render("Predicted Values", True, TEXT_COLOR)
        screen.blit(pred_values_text, (cm_rect.left - pred_values_text.get_width() - 10, cm_rect.top - 30))
        
        cm_labels = [('TP(RpR)', metrics['TP(RpR)']), ('FP(BpR)', metrics['FP(BpR)']), ('FN(RpB)', metrics['FN(RpB)']), ('TN(BpB)', metrics['TN(BpB)'])]
        for i, (label, value) in enumerate(cm_labels):
            x = cm_rect.left + (i % 2) * (cm_rect.width//2) + cm_rect.width//4
            y = cm_rect.top + (i // 2) * (cm_rect.height//2) + cm_rect.height//4
            text = text_font.render(f"{label}: {value}", True, TEXT_COLOR)
            screen.blit(text, (x - text.get_width()//2, y - text.get_height()//2))
        
        # Calculations
        calcs = [
            f"Accuracy = (TP(RpR) + TN(BpB)) / (TP(RpR) + TN(BpB) + FP(BpR) + FN(RpB)) = {metrics['Accuracy']:.4f}",
            f"Precision = TP(RpR) / (TP(RpR) + FP(BpR)) = {metrics['Precision']:.4f}",
            f"Recall = TP(RpR) / (TP(RpR) + FN(RpB)) = {metrics['Recall']:.4f}",
            f"F1-score = 2 * (Precision * Recall) / (Precision + Recall) = {metrics['F1-score']:.4f}"
        ]
        for i, calc in enumerate(calcs):
            text = text_font.render(calc, True, TEXT_COLOR)
            screen.blit(text, (metrics_rect.left + 10, cm_rect.bottom + 20 + i * 30))
    else:
        text = subtitle_font.render("Train the model to see metrics", True, TEXT_COLOR)
        screen.blit(text, (metrics_rect.left + 20, metrics_rect.top + 20))

def draw_button(button, color, text):
    pygame.draw.rect(screen, color, button, border_radius=10)
    pygame.draw.rect(screen, TEXT_COLOR, button, 2, border_radius=10)
    text_surf = text_font.render(text, True, TEXT_COLOR)
    screen.blit(text_surf, (button.centerx - text_surf.get_width() // 2, 
                            button.centery - text_surf.get_height() // 2))

def draw():
    screen.fill(BACKGROUND)
    
    # Draw title and subtitle
    title = title_font.render("Confusion Matrix is no longer Confusing", True, TEXT_COLOR)
    screen.blit(title, (WIDTH // 2 - title.get_width() // 2, 20))
    
    subtitle = subtitle_font.render("Developed by : Venugopal Adep", True, TEXT_COLOR)
    screen.blit(subtitle, (WIDTH // 2 - subtitle.get_width() // 2, 80))
    
    # Draw plot area boundary
    plot_rect = pygame.Rect(0, 100, WIDTH//2, HEIGHT-160)
    pygame.draw.rect(screen, TEXT_COLOR, plot_rect, 2)
    
    # Draw decision boundary and points
    draw_decision_boundary()
    
    if model_trained and model is not None:
        y_pred = model.predict(np.array(points))
        for (x, y), label, pred in zip(points, labels, y_pred):
            color = POSITIVE_COLOR if label == 1 else NEGATIVE_COLOR
            pygame.draw.circle(screen, color, (int(x), int(y)), 7)
            if label == pred:
                text = "TP(RpR)" if label == 1 else "TN(BpB)"
            else:
                text = "FP(BpR)" if pred == 1 else "FN(RpB)"
            text_surf = text_font.render(text, True, TEXT_COLOR)
            screen.blit(text_surf, (int(x) + 10, int(y) - 10))
    else:
        for point, label in zip(points, labels):
            color = POSITIVE_COLOR if label == 1 else NEGATIVE_COLOR
            pygame.draw.circle(screen, color, point, 7)
    
    # Draw metrics
    draw_metrics()
    
    # Draw buttons
    draw_button(clear_button, YELLOW, "Clear")
    draw_button(train_button, GREEN, "Train")
    
    # Draw instructions
    instructions = [
        "Left-click: Add positive point (Red)",
        "Right-click: Add negative point (Blue)",
        "Train: Fit logistic regression model",
        "Clear: Remove all points",
    ]
    
    for i, instruction in enumerate(instructions):
        text = text_font.render(instruction, True, TEXT_COLOR)
        screen.blit(text, (WIDTH//2 + 10, HEIGHT - 140 + i * 30))
    
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
            elif event.pos[0] < WIDTH//2 and 100 < event.pos[1] < HEIGHT-160:
                if event.button == 1:  # Left click
                    add_point(event.pos, 1)  # Positive class (red)
                elif event.button == 3:  # Right click
                    add_point(event.pos, 0)  # Negative class (blue)
    
    draw()

pygame.quit()
