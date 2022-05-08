from time import sleep

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import numpy as np


class Engine:
    DELAY = 0.01  # time between frames in seconds

    DELTA_ALPHA = 1.0
    DELTA_MOVE = 0.1
    DELTA_ZOOM = 0.1

    fov = 90.0  # field of view in angles
    fov_rad = 1.0 / np.tan(fov * 0.5 / 180.0 * np.pi)
    far = 20.0
    near = 1.0
    diff = far - near

    def __init__(self, app_name: str, window_size: tuple[int, int],
                 window_position: tuple[int, int], point_size: float):
        self.app_name = app_name
        self.window = None
        self.window_size = window_size
        self.window_position = window_position
        self.point_size = point_size

        aspect_ratio = float(window_size[1]) / float(window_size[0])  # height / width

        self.projector = np.array([[aspect_ratio * self.fov_rad, 0,            0,                                  0],
                                   [0,                           self.fov_rad, 0,                                  0],
                                   [0,                           0,            self.far / self.diff,               1],
                                   [0,                           0,            - self.far * self.near / self.diff, 0]])

        self.mesh = (
            # Bottom
            np.array([[0.0,  0.0,  0.0],  [0.0,  1.0,  0.0],  [1.0,  1.0,  0.0]]),
            np.array([[0.0,  0.0,  0.0],  [1.0,  1.0,  0.0],  [1.0,  0.0,  0.0]]),

            # Left visible
            np.array([[1.0,  0.0,  0.0],  [1.0,  1.0,  0.0],  [1.0,  1.0,  1.0]]),
            np.array([[1.0,  0.0,  0.0],  [1.0,  1.0,  1.0],  [1.0,  0.0,  1.0]]),

            # Upper
            np.array([[1.0,  0.0,  1.0],  [1.0,  1.0,  1.0],  [0.0,  1.0,  1.0]]),
            np.array([[1.0,  0.0,  1.0],  [0.0,  1.0,  1.0],  [0.0,  0.0,  1.0]]),

            # Right not visible
            np.array([[0.0,  0.0,  1.0],  [0.0,  1.0,  1.0],  [0.0,  1.0,  0.0]]),
            np.array([[0.0,  0.0,  1.0],  [0.0,  1.0,  0.0],  [0.0,  0.0,  0.0]]),

            # Right visible
            np.array([[0.0,  1.0,  0.0],  [0.0,  1.0,  1.0],  [1.0,  1.0,  1.0]]),
            np.array([[0.0,  1.0,  0.0],  [1.0,  1.0,  1.0],  [1.0,  1.0,  0.0]]),

            # Left not visible
            np.array([[1.0,  0.0,  1.0],  [0.0,  0.0,  1.0],  [0.0,  0.0,  0.0]]),
            np.array([[1.0,  0.0,  1.0],  [0.0,  0.0,  0.0],  [1.0,  0.0,  0.0]]),
        )

        self.alpha_X = 0.0
        self.alpha_Z = 0.0

        self.move_X = 0.0
        self.move_Y = 0.0

        self.zoom = 1.0

    def start(self) -> None:
        glutInit()
        glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB)
        glutInitWindowSize(self.window_size[0], self.window_size[1])
        glutInitWindowPosition(self.window_position[0], self.window_position[1])
        self.window = glutCreateWindow(self.app_name)
        self.on_user_create()

        glutDisplayFunc(self.on_user_update)
        glutKeyboardFunc(self.WASD)
        glutSpecialFunc(self.arrows)
        glutMouseFunc(self.mouse)
        glutMouseWheelFunc(self.mouse_wheel)

        glutMainLoop()

    def on_user_create(self) -> None:
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glPointSize(self.point_size)
        gluOrtho2D(0, self.window_size[0], 0, self.window_size[1])

    def on_user_update(self) -> None:
        triangles = []

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        for triangle in self.mesh:
            triangle = self.render_triangle(triangle)
            if triangle is not None:
                triangles.append(triangle)

        triangles = sorted(triangles, key=lambda arr2d: arr2d[0][2] + arr2d[1][2] + arr2d[2][2])

        for triangle in triangles:
            self.draw_triangle(triangle, (0.5, 0.5, 0.5))

        glFlush()

        sleep(self.DELAY)

    def render_triangle(self, triangle: np.array) -> np.array:
        # Rotate by Z
        triangle = self.apply_transformation(self.get_Z_rotation_matrix(self.alpha_Z), triangle)

        # Rotate by X
        triangle = self.apply_transformation(self.get_X_rotation_matrix(self.alpha_X), triangle)

        # Offset into the screen
        offset = 3.0
        for row in range(0, 3):
            triangle[row][2] += offset

        # Move in space
        triangle = self.apply_transformation(self.get_translation_matrix(self.move_X, self.move_Y, 0.0), triangle)

        # Get projection
        triangle = self.apply_transformation(self.get_projection_matrix(), triangle)

        # Zoom in or out
        triangle = self.apply_transformation(self.get_scale_matrix(self.zoom, self.zoom, self.zoom), triangle)

        # Scale into view
        view_scale_1 = 1
        view_scale_2 = 0.5
        for row in range(0, 3):
            for col in range(0, 2):
                triangle[row][col] += view_scale_1
                triangle[row][col] *= view_scale_2 * self.window_size[col]

        if self.triangle_is_visible(triangle):
            self.draw_triangle(triangle, (1, 1, 1))
            return triangle

        return None

    def WASD(self, key, x, y):
        if key == b'a':
            self.alpha_Z += self.DELTA_ALPHA
        if key == b'd':
            self.alpha_Z -= self.DELTA_ALPHA
        if key == b'w':
            self.alpha_X += self.DELTA_ALPHA
        if key == b's':
            self.alpha_X -= self.DELTA_ALPHA
        glutPostRedisplay()

    def arrows(self, key, x, y):
        if key == GLUT_KEY_LEFT:
            self.move_X += self.DELTA_MOVE
        if key == GLUT_KEY_RIGHT:
            self.move_X -= self.DELTA_MOVE
        if key == GLUT_KEY_UP:
            self.move_Y -= self.DELTA_MOVE
        if key == GLUT_KEY_DOWN:
            self.move_Y += self.DELTA_MOVE
        glutPostRedisplay()

    def mouse(self, button, state, x, y):
        pass

    def mouse_wheel(self, wheel, direction, x, y):
        if direction < 0 and self.zoom <= 0.1005:  # restriction on the small size
            return
        self.zoom += direction * self.DELTA_ZOOM
        glutPostRedisplay()

    @staticmethod
    def draw_triangle(triangle: np.array, color: tuple[float, float, float]) -> None:
        glColor3f(color[0], color[1], color[2])
        glBegin(GL_TRIANGLE_STRIP)
        glVertex2f(triangle[0][0], triangle[0][1])
        glVertex2f(triangle[1][0], triangle[1][1])
        glVertex2f(triangle[2][0], triangle[2][1])
        glEnd()

        glColor3f(0.0, 0.0, 0.0)
        glBegin(GL_LINE_STRIP)
        glVertex2f(triangle[0][0], triangle[0][1])
        glVertex2f(triangle[1][0], triangle[1][1])
        glVertex2f(triangle[2][0], triangle[2][1])
        glEnd()

    def apply_transformation(self, t_matrix: np.array, triangle: np.array) -> np.array:
        transformed = np.zeros(shape=(3, 3), dtype=float)

        for idx in range(0, 3):
            homo_vector = self.matrix_4x4_mul_vector_4x1(t_matrix, triangle[idx])
            transformed[idx] = self.vector_from_homo_to_3d(homo_vector)

        return transformed

    def get_projection_matrix(self):
        return self.projector

    @staticmethod
    def get_translation_matrix(T_x: float, T_y: float, T_z: float) -> np.array:
        return np.array([[1.0,  0.0,  0.0,  T_x],
                         [0.0,  1.0,  0.0,  T_y],
                         [0.0,  0.0,  1.0,  T_z],
                         [0.0,  0.0,  0.0,  1.0]])

    @staticmethod
    def get_X_rotation_matrix(degree: float):
        phi = np.deg2rad(degree)
        return np.array([[1.0,      0.0,           0.0,      0.0],
                         [0.0,  np.cos(phi),  -np.sin(phi),  0.0],
                         [0.0,  np.sin(phi),   np.cos(phi),  0.0],
                         [0.0,      0.0,           0.0,      1.0]])

    @staticmethod
    def get_Y_rotation_matrix(degree: float):
        phi = np.deg2rad(degree)
        return np.array([[ np.cos(phi),  0.0,  np.sin(phi),  0.0],
                         [     0.0,      1.0,      0.0,      0.0],
                         [-np.sin(phi),  0.0,  np.cos(phi),  0.0],
                         [     0.0,      0.0,      0.0,      1.0]])

    @staticmethod
    def get_Z_rotation_matrix(degree: float):
        phi = np.deg2rad(degree)
        return np.array([[np.cos(phi),  -np.sin(phi),  0.0,  0.0],
                         [np.sin(phi),   np.cos(phi),  0.0,  0.0],
                         [    0.0,           0.0,      1.0,  0.0],
                         [    0.0,           0.0,      0.0,  1.0]])

    @staticmethod
    def get_scale_matrix(S_x: float, S_y: float, S_z: float) -> np.array:
        return np.array([[S_x,  0.0,  0.0,  0.0],
                         [0.0,  S_y,  0.0,  0.0],
                         [0.0,  0.0,  S_z,  0.0],
                         [0.0,  0.0,  0.0,  1.0]])

    @staticmethod
    def vector_from_3d_to_homo(vector: np.array, w_value=1) -> np.array:
        return np.array([vector[0], vector[1], vector[2], w_value], dtype=object)

    @staticmethod
    def vector_from_homo_to_3d(vector: np.array):
        w = vector[3]
        if w == 0: w = 1
        return np.array([vector[0] / w, vector[1] / w, vector[2] / w])

    @staticmethod
    def matrix_4x4_mul_vector_4x1(matrix_4x4: np.array, vector_3x1: np.array):
        return np.matmul(matrix_4x4, Engine.vector_from_3d_to_homo(vector_3x1))

    @staticmethod
    def triangle_is_visible(triangle: np.array):
        edge_1 = triangle[1] - triangle[0]
        edge_2 = triangle[2] - triangle[0]

        normal = np.cross(edge_1, edge_2)
        normal / np.linalg.norm(normal)

        return normal[2] < 0


if __name__ == "__main__":
    demo = Engine("3d_demo", (500, 500), (100, 100), 10.0)
    demo.start()
