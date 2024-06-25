import pygame
from pygame.math import Vector3
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Initialize Pygame and OpenGL
pygame.init()
WIDTH, HEIGHT = 1600, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.DOUBLEBUF | pygame.OPENGL)
pygame.display.set_caption("Kernel Trick Demo")

# Colors
WHITE = (1, 1, 1)
BLACK = (0, 0, 0)
RED = (1, 0, 0)
BLUE = (0, 0, 1)
GREEN = (0, 1, 0)
YELLOW = (1, 1, 0)
PURPLE = (0.5, 0, 0.5)
DARK_BLUE = (0, 0, 0.5)
ORANGE = (1, 0.65, 0)

# Fonts
title_font = pygame.font.Font(None, 64)
subtitle_font = pygame.font.Font(None, 36)
font = pygame.font.Font(None, 24)

# Data
points_2d = []
labels = []

# SVM model
model = None
scaler = StandardScaler()

# Camera
camera_distance = 5
camera_angle_x = 30
camera_angle_y = 45

# Buttons
apply_button = pygame.Rect(WIDTH - 220, HEIGHT - 60, 100, 40)
reset_button = pygame.Rect(WIDTH - 110, HEIGHT - 60, 100, 40)

def draw_text_2d(text, font, x, y, color):
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, (x, y))

def add_point(pos, label):
    points_2d.append(pos)
    labels.append(label)

def reset_data():
    global points_2d, labels, model
    points_2d = []
    labels = []
    model = None

def apply_kernel():
    global model
    if len(points_2d) > 1 and len(set(labels)) > 1:
        X = np.array(points_2d)
        y = np.array(labels)
        X_scaled = scaler.fit_transform(X)
        model = SVC(kernel='rbf')
        model.fit(X_scaled, y)

def transform_to_3d(point_2d):
    x, y = point_2d
    z = x**2 + y**2  # Simple quadratic transformation
    return Vector3(x, y, z)

def draw_points():
    glPointSize(10)
    glBegin(GL_POINTS)
    for point, label in zip(points_2d, labels):
        color = RED if label == 0 else BLUE
        glColor3fv(color)
        point_3d = transform_to_3d(point)
        glVertex3f(point_3d.x, point_3d.y, point_3d.z)
    glEnd()

def draw_axes():
    glBegin(GL_LINES)
    # X-axis (red)
    glColor3f(1, 0, 0)
    glVertex3f(0, 0, 0)
    glVertex3f(1, 0, 0)
    # Y-axis (green)
    glColor3f(0, 1, 0)
    glVertex3f(0, 0, 0)
    glVertex3f(0, 1, 0)
    # Z-axis (blue)
    glColor3f(0, 0, 1)
    glVertex3f(0, 0, 0)
    glVertex3f(0, 0, 1)
    glEnd()

def draw_separating_surface():
    if model is not None:
        xx, yy = np.meshgrid(np.linspace(-1, 1, 20), np.linspace(-1, 1, 20))
        xy = np.vstack([xx.ravel(), yy.ravel()]).T
        zz = model.decision_function(scaler.transform(xy)).reshape(xx.shape)
        
        glBegin(GL_QUADS)
        glColor4f(*PURPLE, 0.5)  # Semi-transparent purple
        for i in range(19):
            for j in range(19):
                glVertex3f(xx[i, j], yy[i, j], zz[i, j])
                glVertex3f(xx[i+1, j], yy[i+1, j], zz[i+1, j])
                glVertex3f(xx[i+1, j+1], yy[i+1, j+1], zz[i+1, j+1])
                glVertex3f(xx[i, j+1], yy[i, j+1], zz[i, j+1])
        glEnd()

def draw():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    
    # Set up camera
    gluPerspective(45, (WIDTH / HEIGHT), 0.1, 50.0)
    glTranslatef(0.0, 0.0, -camera_distance)
    glRotatef(camera_angle_x, 1, 0, 0)
    glRotatef(camera_angle_y, 0, 1, 0)

    # Draw 3D scene
    draw_axes()
    draw_points()
    draw_separating_surface()

    # Switch to 2D rendering for UI elements
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, WIDTH, HEIGHT, 0, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    # Clear the depth buffer to ensure 2D elements are drawn on top
    glClear(GL_DEPTH_BUFFER_BIT)

    # Draw 2D UI elements
    screen.fill((255, 255, 255, 0), special_flags=pygame.BLEND_RGBA_MIN)

    # Draw title and subtitle
    draw_text_2d("Kernel Trick Demo", title_font, WIDTH // 2 - 200, 30, DARK_BLUE)
    draw_text_2d("Developed by: Venugopal Adep", subtitle_font, WIDTH // 2 - 180, 80, DARK_BLUE)

    # Draw instructions
    instructions = [
        "Left-click: Add red point (Class 0)",
        "Right-click: Add blue point (Class 1)",
        "Drag: Rotate view",
        "Scroll: Zoom in/out"
    ]
    for i, instruction in enumerate(instructions):
        draw_text_2d(instruction, font, 10, HEIGHT - 120 + i * 30, DARK_BLUE)

    # Draw buttons
    pygame.draw.rect(screen, GREEN, apply_button)
    pygame.draw.rect(screen, ORANGE, reset_button)
    draw_text_2d("Apply Kernel", font, WIDTH - 210, HEIGHT - 50, BLACK)
    draw_text_2d("Reset", font, WIDTH - 90, HEIGHT - 50, BLACK)

    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)

    pygame.display.flip()

def main():
    global camera_angle_x, camera_angle_y, camera_distance
    
    clock = pygame.time.Clock()
    dragging = False
    last_pos = None
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1 or event.button == 3:  # Left or right click
                    x, y = event.pos
                    if x < WIDTH // 2:  # Only add points in the left half
                        normalized_x = (x - WIDTH/4) / (WIDTH/4)
                        normalized_y = (HEIGHT/2 - y) / (HEIGHT/2)
                        add_point((normalized_x, normalized_y), 0 if event.button == 1 else 1)
                elif event.button == 4:  # Scroll up
                    camera_distance = max(camera_distance - 0.1, 2)
                elif event.button == 5:  # Scroll down
                    camera_distance = min(camera_distance + 0.1, 10)
                dragging = True
                last_pos = event.pos
            elif event.type == pygame.MOUSEBUTTONUP:
                if apply_button.collidepoint(event.pos):
                    apply_kernel()
                elif reset_button.collidepoint(event.pos):
                    reset_data()
                dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if dragging and last_pos:
                    dx, dy = event.pos[0] - last_pos[0], event.pos[1] - last_pos[1]
                    camera_angle_y += dx * 0.5
                    camera_angle_x += dy * 0.5
                    last_pos = event.pos

        draw()
        clock.tick(60)

if __name__ == "__main__":
    main()