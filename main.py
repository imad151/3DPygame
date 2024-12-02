import random
import pygame
import math
import numpy as np
from scipy.spatial import ConvexHull

class ShapeCanvas:
    def __init__(self):
        self.screen_width = 800
        self.screen_height = 600
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("3D Shape Renderer")

        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)

        # Shape vertices and edges
        self.vertices, self.edges = self.GetRandomShape()

        # Size
        self.scale = 300

        # Cam distance
        self.cam_distance = 5

        # Rotation angles
        self.angle_x = 0
        self.angle_y = 0

        self.clock = pygame.time.Clock()

    def project(self, point):
        """
        Project a 3D point to a 2D screen.
        x' = x / (z + d) * s
        d = distance from cam
        s = scale
        """
        scale = self.scale
        cam_distance = self.cam_distance
        x, y, z = point
        factor = scale / (z + cam_distance)
        x_proj = x * factor + self.screen_width // 2
        y_proj = -y * factor + self.screen_height // 2
        return int(x_proj), int(y_proj)

    def rotate(self, point, angle_x, angle_y):
        """
        Rotate a 3D point around the X and Y axes.
        """
        x, y, z = point

        # Rotate around X axis
        cos_x = math.cos(angle_x)
        sin_x = math.sin(angle_x)
        y, z = y * cos_x - z * sin_x, y * sin_x + z * cos_x

        # Rotate around Y axis
        cos_y = math.cos(angle_y)
        sin_y = math.sin(angle_y)
        x, z = x * cos_y + z * sin_y, -x * sin_y + z * cos_y

        return x, y, z

    def GetRandomShape(self):
        """
        Generate a random 3D shape based on random points and compute its convex hull.
        """
        num_points = random.randint(10, 20)
        points = np.random.rand(num_points, 3) * 2 - 1

        hull = ConvexHull(points)

        edges = []

        for simplex in hull.simplices:
            edges.append((simplex[0], simplex[1]))
            edges.append((simplex[1], simplex[2]))
            edges.append((simplex[2], simplex[0]))

        return points, edges

    def draw_shape(self):
        """
        Draw the shape on the screen.
        """
        transformed_points = [self.rotate(v, self.angle_x, self.angle_y) for v in self.vertices]
        projected_points = [self.project(p) for p in transformed_points]

        for edge in self.edges:
            start, end = edge

            # Check if the indices are valid
            if start >= len(projected_points) or end >= len(projected_points):
                print(f"Invalid edge: {edge} (out of range)")
                continue  # Skip invalid edges

            pygame.draw.line(self.screen, self.WHITE, projected_points[start], projected_points[end], 2)

    def handle_mouse_input(self):
        """Update rotation angles based on mouse movement."""
        mouse_dx, mouse_dy = pygame.mouse.get_rel()
        self.angle_x += math.radians(mouse_dy) * 0.2  # sensitivity for pitch
        self.angle_y += math.radians(mouse_dx) * 0.2  # sensitivity for yaw

    def game_loop(self):
        """Main game loop."""
        running = True

        # Enable mouse control
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            self.screen.fill(self.BLACK)

            self.handle_mouse_input()
            self.draw_shape()
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()


if __name__ == "__main__":
    pygame.init()
    shape_canvas = ShapeCanvas()
    shape_canvas.game_loop()
