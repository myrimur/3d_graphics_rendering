from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import numpy as np


class Engine:
    fov = 90.0  # field of view in angles
    fov_rad = 1.0 / np.tan(fov * 0.5 / 180.0 * np.pi)
    far = 20.0
    near = 1.0
    diff = far - near

    def __init__(self, app_name: str, window_size: tuple[int, int],
                 window_position: tuple[int, int], point_size: float):
        self.app_name = app_name
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

    def start(self) -> None:
        glutInit()
        glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB)
        glutInitWindowSize(self.window_size[0], self.window_size[1])
        glutInitWindowPosition(self.window_position[0], self.window_position[1])
        glutCreateWindow(self.app_name)
        self.on_user_create()
        glutDisplayFunc(self.on_user_update)
        glutMainLoop()

    def on_user_create(self) -> None:
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glPointSize(self.point_size)
        gluOrtho2D(0, self.window_size[0], 0, self.window_size[1])

    def on_user_update(self) -> None:
        glClear(GL_COLOR_BUFFER_BIT)

        s_coef = 1  # saturation scaler
        for triangle in self.mesh:

            # Offset into the screen
            offset = 3.0
            for row in range(0, 3):
                triangle[row][2] += offset

            # Get projection
            triangle = self.apply_transformation(self.get_projection_matrix(), triangle)

            # Scale into view
            view_scale_1 = 1
            view_scale_2 = 0.5

            for row in range(0, 3):
                for col in range(0, 2):
                    triangle[row][col] += view_scale_1
                    triangle[row][col] *= view_scale_2 * self.window_size[col]

            self.draw_triangle(triangle, (0.1 * s_coef, 0.1 * s_coef, 0.1 * s_coef))
            s_coef += 1

        glFlush()

    @staticmethod
    def draw_triangle(triangle: np.array, color: tuple[float, float, float]) -> None:
        glColor3f(color[0], color[1], color[2])
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)
        glBegin(GL_TRIANGLE_STRIP)
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


if __name__ == "__main__":
    demo = Engine("3d_demo", (500, 500), (100, 100), 10.0)
    demo.start()
