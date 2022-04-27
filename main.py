from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import numpy as np


class Engine:
    def __init__(self, app_name: str, window_size: tuple[int, int],
                 window_position: tuple[int, int], point_size: float):
        self.app_name = app_name
        self.window_size = window_size
        self.window_position = window_position
        self.point_size = point_size

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
        glColor3f(1.0, 1.0, 1.0)
        glPointSize(self.point_size)
        gluOrtho2D(0, self.window_size[0], 0, self.window_size[1])

    def on_user_update(self) -> None:
        glClear(GL_COLOR_BUFFER_BIT)

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
    def draw_triangle(triangle: tuple[np.array]) -> None:
        glBegin(GL_TRIANGLE_STRIP)
        glVertex2f(triangle[0][0], triangle[0][1])
        glVertex2f(triangle[1][0], triangle[1][1])
        glVertex2f(triangle[2][0], triangle[2][1])
        glEnd()


if __name__ == "__main__":
    demo = Engine("3d_demo", (500, 500), (100, 100), 10.0)
    demo.start()
