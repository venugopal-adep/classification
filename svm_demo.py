import pygame
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Initialize Pygame
pygame.init()

# Set up the display
width, height = 1600, 800
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("SVM Demo with Metrics")

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)

# Font
font = pygame.font.Font(None, 36)

# SVM parameters
X = []
y = []
X_train, X_test, y_train, y_test = [], [], [], []
clf = svm.SVC(kernel='linear')

def add_point(pos, class_label):
    X.append([pos[0], pos[1]])
    y.append(class_label)

def train_svm():
    global X_train, X_test, y_train, y_test
    if len(X) > 1 and len(set(y)) > 1:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf.fit(X_train, y_train)

def draw_points():
    for i, point in enumerate(X):
        color = RED if y[i] == 0 else BLUE
        pygame.draw.circle(screen, color, (int(point[0]), int(point[1])), 10)

def draw_svm_boundary():
    if len(X) > 1 and len(set(y)) > 1:
        x_min, x_max = 0, width
        y_min, y_max = 0, height

        XX, YY = np.meshgrid(np.arange(x_min, x_max, 10),
                             np.arange(y_min, y_max, 10))
        Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
        Z = Z.reshape(XX.shape)

        # Draw the decision boundary
        for i in range(len(XX)):
            for j in range(len(XX[i]) - 1):
                if Z[i][j] * Z[i][j+1] <= 0:
                    pygame.draw.line(screen, GREEN, 
                                     (int(XX[i][j]), int(YY[i][j])),
                                     (int(XX[i][j+1]), int(YY[i][j+1])), 2)

        for i in range(len(XX) - 1):
            for j in range(len(XX[i])):
                if Z[i][j] * Z[i+1][j] <= 0:
                    pygame.draw.line(screen, GREEN, 
                                     (int(XX[i][j]), int(YY[i][j])),
                                     (int(XX[i+1][j]), int(YY[i+1][j])), 2)

def calculate_metrics():
    if len(X_test) > 0:
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        return accuracy, precision, recall, f1
    return None

def draw_text(text, position, color=WHITE):
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, position)

# Main game loop
running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            pos = pygame.mouse.get_pos()
            if event.button == 1:  # Left click
                add_point(pos, 0)
            elif event.button == 3:  # Right click
                add_point(pos, 1)
            train_svm()

    screen.fill(BLACK)

    draw_points()
    draw_svm_boundary()

    # Draw labels
    draw_text("Left Click: Class 1 (Red)", (10, 10), RED)
    draw_text("Right Click: Class 2 (Blue)", (10, 50), BLUE)
    draw_text("Green line: SVM Decision Boundary", (10, 90), GREEN)

    # Calculate and display metrics
    metrics = calculate_metrics()
    if metrics:
        accuracy, precision, recall, f1 = metrics
        draw_text(f"Accuracy: {accuracy:.2f}", (10, height - 160), WHITE)
        draw_text(f"Precision: {precision:.2f}", (10, height - 120), WHITE)
        draw_text(f"Recall: {recall:.2f}", (10, height - 80), WHITE)
        draw_text(f"F1-score: {f1:.2f}", (10, height - 40), WHITE)

    pygame.display.flip()
    clock.tick(30)

pygame.quit()