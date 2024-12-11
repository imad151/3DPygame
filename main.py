import pygame
import math
import numpy as np
import matplotlib.pyplot as plt
from ConvexHullImplementation.PlottingConvexHull import ConvexHull3D

class ShapeCanvas:
    def __init__(self):
        self.screen_width = 1920
        self.screen_height = 1080
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("3D Shape Renderer")

        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 0, 0)
        self.GRAY = (128, 128, 128)
        self.BLUE = (0, 0, 255)

        self.InitShape()
        self.InitTextBoxes()

        self.scale = 100
        self.cam_distance = 15
        self.cam_pos = np.array([0, 0, self.cam_distance])

        self.angle_x = 0
        self.angle_y = 0

        self.clock = pygame.time.Clock()

    def InitShape(self):
        self.vertices, self.edges = self.GetRandomShape()
        self.center_point = np.mean(self.vertices, axis=0)  # Center point for volume calculation
        self.faces = self.calculate_faces()  # Calculate faces for volume computation

        self.volume = self.compute_polyhedron_volume()
        print(f"Polyhedron Volume: {self.volume}")

    def InitTextBoxes(self):
        self.font = pygame.font.Font(None, 36)
        self.text_surfaces = [
            self.font.render("Move: Mouse", True, self.WHITE),
            self.font.render("New Figure: Enter", True, self.WHITE),
            self.font.render("Quit: Escape", True, self.WHITE),
            self.font.render(f"Volume of Figure: {self.volume:.2f}", True, self.WHITE)
        ]

    def project(self, point):
        scale = self.scale
        cam_distance = self.cam_distance
        x, y, z = point
        factor = scale / (z + cam_distance + 1e-6)

        x_proj = x * factor + self.screen_width // 2
        y_proj = -y * factor + self.screen_height // 2
        return int(x_proj), int(y_proj), z

    def rotate(self, point, angle_x, angle_y, angle_z=0):
        centroid = np.mean(self.vertices, axis=0)

        x, y, z = point - centroid

        cos_x, sin_x = math.cos(angle_x), math.sin(angle_x)
        y, z = y * cos_x - z * sin_x, y * sin_x + z * cos_x

        cos_y, sin_y = math.cos(angle_y), math.sin(angle_y)
        x, z = x * cos_y + z * sin_y, -x * sin_y + z * cos_y

        cos_z, sin_z = math.cos(angle_z), math.sin(angle_z)
        x, y = x * cos_z - y * sin_z, x * sin_z + y * cos_z

        rotated_point = np.array([x, y, z]) + centroid
        return rotated_point

    def GetRandomShape(self):
        num_points = np.random.randint(5, 10)
        points = []
        min_distance = 7

        for _ in range(num_points):
            while True:
                x = np.random.uniform(-10, 10)
                y = np.random.uniform(-10, 10)
                z = np.random.uniform(-10, 10)
                new_point = np.array([x, y, z])

                too_close = False
                for point in points:
                    if np.linalg.norm(new_point - point) < min_distance:
                        too_close = True
                        break

                if not too_close:
                    points.append(new_point)
                    break

        points = np.array(points)

        hull = ConvexHull3D(points)
        hull_edges = hull.getEdges()

        point_to_index = {tuple(point): idx for idx, point in enumerate(points)}

        edges = []
        for start, end in hull_edges:
            start_idx = point_to_index.get(tuple(start))
            end_idx = point_to_index.get(tuple(end))
            if start_idx is not None and end_idx is not None:
                edges.append((start_idx, end_idx))

        return points, edges

    def calculate_faces(self):
        # Derive triangular faces from edges for volume calculation.
        faces = []
        for i in range(0, len(self.edges), 3):
            indices = self.edges[i:i + 3]
            if len(indices) == 3:
                faces.append([start for start, end in indices])
        return faces

    def compute_tetrahedron_volume(self, p1, p2, p3, p4):
        matrix = np.array([
            [p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]],
            [p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]],
            [p4[0] - p1[0], p4[1] - p1[1], p4[2] - p1[2]],
        ])
        return abs(np.linalg.det(matrix)) / 6

    def compute_polyhedron_volume(self):
        total_volume = 0
        for face in self.faces:
            if len(face) == 3:
                p1 = self.center_point
                p2, p3, p4 = self.vertices[face]
                volume = self.compute_tetrahedron_volume(p1, p2, p3, p4)
                total_volume += volume
        return total_volume

    def draw_shape(self):
        transformed_points = [self.rotate(v, self.angle_x, self.angle_y) for v in self.vertices.tolist()]
        projected_points = [self.project(p) for p in transformed_points]

        faces = []
        for i in range(0, len(self.edges), 3):
            indices = self.edges[i:i + 3]
            if len(indices) < 3:
                continue

            vertices = np.array([transformed_points[start] for start, end in indices])
            projected = [projected_points[start][:2] for start, end in indices]
            depth = np.mean([transformed_points[start][2] for start, end in indices])

            faces.append((depth, projected, vertices))

        # Sort faces by depth (Painter's Algorithm)
        faces.sort(key=lambda x: x[0], reverse=True)

        for _, projected, _ in faces:
            pygame.draw.polygon(self.screen, self.GRAY, projected)

        # Draw edges on top
        for edge in self.edges:
            start, end = edge
            if start >= len(projected_points) or end >= len(projected_points):
                continue
            pygame.draw.line(self.screen, self.RED, projected_points[int(start)][:2], projected_points[int(end)][:2], 3)

        # Draw centroid
        rotated_centroid = self.rotate(self.center_point, self.angle_x, self.angle_y)
        centroid_proj = self.project(rotated_centroid)
        centroid_screen = centroid_proj[:2]
        size = max(1, int(10 / (centroid_proj[2] + self.cam_distance + 1e-6)))
        pygame.draw.circle(self.screen, self.BLUE, centroid_screen, 3)

    def draw_text(self):
        y_offset = 10  # Starting y-coordinate
        for i, text_surface in enumerate(self.text_surfaces):
            self.screen.blit(text_surface, (10, y_offset))
            y_offset += 40  # Move down for the next line

    def handle_mouse_input(self):
        mouse_dx, mouse_dy = pygame.mouse.get_rel()
        rotation_speed = 0.2
        self.angle_x += math.radians(mouse_dy) * rotation_speed
        self.angle_y += math.radians(mouse_dx) * rotation_speed

    def game_loop(self):
        running = True
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)

        while running:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

                    if event.key == pygame.K_RETURN:
                        self.InitShape()

            self.screen.fill(self.BLACK)
            self.handle_mouse_input()
            self.draw_shape()
            self.draw_text()
            pygame.display.flip()
            self.clock.tick(60)

        pygame.quit()


if __name__ == "__main__":
    pygame.init()
    try:
        shape_canvas = ShapeCanvas()

        # Matplotlib 3D plot for testing
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        vertices = shape_canvas.vertices
        edges = shape_canvas.edges

        for edge in edges:
            start, end = edge
            line = np.array([vertices[start], vertices[end]])
            ax.plot(line[:, 0], line[:, 1], line[:, 2], color="b")

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()

        shape_canvas.game_loop()
    finally:
        pygame.quit()
