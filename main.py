from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import numpy as np


class Engine:
    fov = 90.0  # field of view in angles
    fov_rad = 1 / np.tan(fov * 0.5 / 180 * np.pi)
    far = 1000.0
    near = 0.1
    diff = far - near

    def __init__(self, app_name: str, window_size: tuple[int, int],
                 window_position: tuple[int, int], point_size: float):
        self.app_name = app_name
        self.window_size = window_size
        self.window_position = window_position
        self.point_size = point_size

        aspect_ratio = float(window_size[0]) / float(window_size[1])

        self.projector = np.array([[aspect_ratio * self.fov_rad, 0,            0,                                  0],
                                   [0,                           self.fov_rad, 0,                                  0],
                                   [0,                           0,            self.far / self.diff,               1],
                                   [0,                           0,            - self.far * self.near / self.diff, 0]])

        self.mesh = (
            np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
            np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]),
            np.array([[1.0, 0.0, 0.0], [1.0, 1.0, 1.0], [1.0, 0.0, 1.0]]),
            np.array([[1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]]),
            np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]),
            np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 1.0, 0.0]]),
            np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
            np.array([[0.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]]),
            np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [1.0, 1.0, 0.0]]),
            np.array([[1.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]),
            np.array([[1.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
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
            translated = np.array([np.copy(triangle[0]),
                                   np.copy(triangle[1]),
                                   np.copy(triangle[2])])

            offset = -1
            for row in range(0, 3):
                translated[row][2] += offset

            projected = self.get_projection(translated)

            view_scale_1 = 1.0
            view_scale_2 = 0.5

            for row in range(0, 3):
                for col in range(0, 2):
                    projected[row][col] += view_scale_1
                    projected[row][col] *= view_scale_2 * self.window_size[col]

            self.draw_triangle(projected, (0.1 * s_coef, 0.1 * s_coef, 0.1 * s_coef))
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

    def get_projection(self, triangle: np.array) -> np.array:
        projection = np.zeros(shape=(3, 3), dtype=float)

        for idx in range(0, 3):
            projected = np.matmul( self.projector, np.array([triangle[idx][0], triangle[idx][1], triangle[idx][2], 1]) )
            scalar = projected[-1]
            if scalar == 0: scalar = 1
            projection[idx] = np.array([ projected[0]/scalar, projected[1]/scalar, projected[2]/scalar ])

        return projection


if __name__ == "__main__":
    demo = Engine("3d_demo", (500, 500), (100, 100), 10.0)
    demo.start()
