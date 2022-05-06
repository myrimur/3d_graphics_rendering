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

        self.draw_triangle(
            np.array([[100.0, 210.0], [300.0, 210.0], [300.0, 310.0]]),
            (0.2, 0.5, 0.4))

        glBegin(GL_POINTS)
        glVertex2f(100, 100)
        glVertex2f(300, 200)
        glEnd()

        glBegin(GL_QUADS)
        glVertex2f(100.0, 100.0)
        glVertex2f(300.0, 100.0)
        glVertex2f(300.0, 200.0)
        glVertex2f(100.0, 200.0)
        glEnd()

        glFlush()

    @staticmethod
    def draw_triangle(triangle: np.array, color: tuple[float, float, float]) -> None:
        glColor3f(color[0], color[1], color[2])
        glBegin(GL_TRIANGLE_STRIP)
        glVertex2f(triangle[0][0], triangle[0][1])
        glVertex2f(triangle[1][0], triangle[1][1])
        glVertex2f(triangle[2][0], triangle[2][1])
        glEnd()


if __name__ == "__main__":
    demo = Engine("3d_demo", (500, 500), (100, 100), 10.0)
    demo.start()
