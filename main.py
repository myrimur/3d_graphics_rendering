"""Module with the engine to render and transform the 3D object in 2D space."""

from time import sleep

from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
import numpy as np


class Engine:
    """Engine to render and transform the 3D object in 2D space."""

    DELAY = 0.01  # time between frames in seconds

    DELTA_ALPHA = 1.0  # object rotation angle on WASD
    DELTA_PHI = 30.0  # light rotation angle on click
    DELTA_MOVE = 0.1  # screen move on arrows
    DELTA_ZOOM = 0.1  # object zoom on wheel

    fov = 90.0  # field of view in angles
    fov_rad = 1.0 / np.tan(fov * 0.5 / 180.0 * np.pi)
    far = 20.0  # define where to display in the back by Z
    near = 1.0  # define the front by Z

    def __init__(self, app_name: str, window_size: tuple[int, int],
                 window_position: tuple[int, int], point_size: float,
                 color: tuple[int, int, int], obj_path: str):
        """
        Initialize the required configurations to use OpenGL, and the
        variables corresponding to object transformations and their states in a space.

        :param app_name: The window title
        :param window_size: A tuple with (width, height) size of the window
        :param window_position: A tuple with (x, y) coordinates of the window on screen
                                with the origin at upper left corner
        :param point_size: Float value of a point size
        :param color: A tuple (R, G, B) representing an RGB color as ints from 0 to 255.
        :param obj_path: A path to the .obj file containing information about vertices and
                         faces of the object. Note that the faces should be represented as
                         a triangles.
        """
        self.app_name = app_name
        self.window_size = window_size
        self.window_position = window_position
        self.point_size = point_size
        self.window = None

        # Scale colors from RGB as ints into [0, 1] floats
        self.color = (color[0] / 255.0, color[1] / 255.0, color[2] / 255.0)

        # Define projection matrix onto screen
        aspect_ratio = float(window_size[1]) / float(window_size[0])  # height / width
        diff = self.far - self.near
        self.projector = np.array([[aspect_ratio * self.fov_rad, 0,            0,                             0],
                                   [0,                           self.fov_rad, 0,                             0],
                                   [0,                           0,            self.far / diff,               1],
                                   [0,                           0,            - self.far * self.near / diff, 0]])

        self.mesh = self.read_obj_file(obj_path)  # Read mesh from input .obj file

        self.alpha_X = 0.0  # Object rotation angle by X
        self.alpha_Z = 0.0  # Object rotation angle by Z

        self.move_X = 0.0  # View position by X
        self.move_Y = 0.0  # View position by Z

        self.zoom = 1.0  # Object size

        self.light_phi = 0.0  # Horizontal light direction rotation angle

        self.camera = np.array([0.0, 0.0, 0.0])  # Camera direction
        self.light_direction = np.array([0.0, 0.0, -1.0])  # Light direction

    @staticmethod
    def read_obj_file(path: str) -> tuple[np.array, ...]:
        """
        Parse the .obj file with the given path.
        Finds only vertices and faces given as triangles.
        Returns the tuple of a 2D np.array each representing
        a particular triangle with 3 points.
        """
        vertices = []  # store vertices
        mesh = []  # store triangles

        with open(path) as obj:
            for line in obj:
                line = line.strip().split()
                if len(line) == 0: continue

                if line[0] == 'v':  # it is vertex
                    vertices.append(np.array([float(line[1]), float(line[2]), float(line[3])]))

                elif line[0] == 'f':  # it is face
                    if '/' in line[1]:  # the vertex information is: idx//norm
                        for idx in range(1, 4):
                            line[idx] = line[idx][:line[idx].index('/')]  # ignore norm
                    mesh.append(np.array([vertices[int(line[1]) - 1], vertices[int(line[2]) - 1],
                                          vertices[int(line[3]) - 1]]))
        return tuple(mesh)

    def start(self) -> None:
        """
        Main runner. Initialized and created window,
        proceeds with infinite loop of iterations to
        display and perform transformations from user input.
        """
        # Initialize window configurations
        glutInit()
        glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB)
        glutInitWindowSize(self.window_size[0], self.window_size[1])
        glutInitWindowPosition(self.window_position[0], self.window_position[1])
        self.window = glutCreateWindow(self.app_name)
        self.__on_user_create()

        # Loop begins here
        # Perform object rendering
        glutDisplayFunc(self.__on_user_update)

        # Track input from keyboard and mouse
        glutKeyboardFunc(self.__WASD)
        glutSpecialFunc(self.__arrows)
        glutMouseFunc(self.__mouse)
        glutMouseWheelFunc(self.__mouse_wheel)

        # Enter OpenGL event processing loop
        glutMainLoop()

    def __on_user_create(self) -> None:
        """Define 2D objects visualisation configuration."""
        glClearColor(0.0, 0.0, 0.0, 0.0)
        glPointSize(self.point_size)
        gluOrtho2D(0, self.window_size[0], 0, self.window_size[1])

    def __on_user_update(self) -> None:
        """
        To be called infinitely in the loop. Computes all the transformations
        and renders the object piece-by-piece.
        """
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        triangles = []
        for triangle in self.mesh:
            pair = self.__project_triangle(triangle)
            if pair is not None:  # if triangle is visible
                triangles.append(pair)

        # Sort visible triangles by mean of Z coord
        triangles = sorted(triangles, key=lambda tup: tup[0][0][2] + tup[0][1][2] + tup[0][2][2])

        for triangle, mul in triangles:
            self.draw_triangle(triangle,
                               (self.color[0] * mul, self.color[1] * mul, self.color[2] * mul))

        glFlush()  # force to render triangles
        sleep(self.DELAY)  # wait until next frame

    def __project_triangle(self, triangle: np.array) -> tuple[np.array, float] or None:
        """
        Compute the projection of the triangle into 2D screen. Decide whether
        the triangle should be visible. If so, compute all the transformations,
        and return the tuple of a final triangle representations in 2D
        and its color scaled for the depth. Return None otherwise.
        """
        # Rotate by Z
        triangle = self.__apply_transformation(self.get_Z_rotation_matrix(self.alpha_Z), triangle)

        # Rotate by X
        triangle = self.__apply_transformation(self.get_X_rotation_matrix(self.alpha_X), triangle)

        # Offset into the screen
        offset = 3.0
        for row in range(0, 3):
            triangle[row][2] += offset

        # Move in space
        triangle = self.__apply_transformation(
            self.get_translation_matrix(self.move_X, self.move_Y, 0.0), triangle)

        normal = self.get_normal(triangle)

        # If dot product of triangle's normal and camera-triangle direction is positive,
        # then triangle is not visible
        if np.dot(normal, triangle[0] - self.camera) > 0:
            return None

        # Rotate light direction
        light_direction = self.vector_from_homo_to_3d(
            self.matrix_4x4_mul_vector_4x1(self.get_Y_rotation_matrix(
                self.light_phi), self.light_direction))

        # Illumination
        # Dot product is in range [-1, 1] because vectors are normalized
        dot_product = np.dot(normal, light_direction)
        color = self.get_color_scaler(float(dot_product))  # compute scaler for shadows

        # Get projection
        triangle = self.__apply_transformation(self.get_projection_matrix(), triangle)

        # Zoom in or out
        triangle = self.__apply_transformation(
            self.get_scale_matrix(self.zoom, self.zoom, self.zoom), triangle)

        # Scale into view
        view_scale_1 = 1
        view_scale_2 = 0.5
        for row in range(0, 3):
            for col in range(0, 2):
                triangle[row][col] += view_scale_1
                triangle[row][col] *= view_scale_2 * self.window_size[col]

        return triangle, color

    @staticmethod
    def get_color_scaler(dp: float) -> float:
        """Transform to get the color scaler in range [0.2, 1.0] for the shadows."""
        return (1.5 + dp) / 2.5

    def __WASD(self, key, x, y) -> None:
        """Track keyboard W-A-S-D buttons to rotate the object in space."""
        if key == b'a':
            self.alpha_Z += self.DELTA_ALPHA
        if key == b'd':
            self.alpha_Z -= self.DELTA_ALPHA
        if key == b'w':
            self.alpha_X += self.DELTA_ALPHA
        if key == b's':
            self.alpha_X -= self.DELTA_ALPHA
        glutPostRedisplay()

    def __arrows(self, key, x, y) -> None:
        """Track keyboard arrows buttons to move the screen view around the object."""
        if key == GLUT_KEY_LEFT:
            self.move_X += self.DELTA_MOVE
        if key == GLUT_KEY_RIGHT:
            self.move_X -= self.DELTA_MOVE
        if key == GLUT_KEY_UP:
            self.move_Y -= self.DELTA_MOVE
        if key == GLUT_KEY_DOWN:
            self.move_Y += self.DELTA_MOVE
        glutPostRedisplay()

    def __mouse(self, button, state, x, y) -> None:
        """Track the left and right clicks of the mouse to rotate the light direction."""
        if state == GLUT_DOWN and button == GLUT_LEFT_BUTTON:
            self.light_phi -= self.DELTA_PHI
        elif state == GLUT_DOWN and button == GLUT_RIGHT_BUTTON:
            self.light_phi += self.DELTA_PHI
        glutPostRedisplay()

    def __mouse_wheel(self, wheel, direction, x, y) -> None:
        """Track the mouse wheel for scaling the size of object in space."""
        if direction < 0 and self.zoom <= 0.1005:  # restriction on the small size
            return
        self.zoom += direction * self.DELTA_ZOOM
        glutPostRedisplay()

    @staticmethod
    def draw_triangle(triangle: np.array, color: tuple[float, float, float]) -> None:
        """Visualize the filled triangle."""
        glColor3f(color[0], color[1], color[2])
        glBegin(GL_TRIANGLE_STRIP)
        glVertex2f(triangle[0][0], triangle[0][1])
        glVertex2f(triangle[1][0], triangle[1][1])
        glVertex2f(triangle[2][0], triangle[2][1])
        glEnd()

    def __apply_transformation(self, t_matrix: np.array, triangle: np.array) -> np.array:
        """
        Multiply initial triangle given as 3x3 matrix by the transformation matrix t_matrix
        in homogeneous coordinates with 4x4 shape. It transforms each point in the triangle
        to homogeneous coordinates, performs transformations, and casts back to 3D vector.
        Returns the transformed 3x3 matrix as 2D np.array.
        """
        transformed = np.zeros(shape=(3, 3), dtype=float)

        for idx in range(0, 3):
            homo_vector = self.matrix_4x4_mul_vector_4x1(t_matrix, triangle[idx])
            transformed[idx] = self.vector_from_homo_to_3d(homo_vector)

        return transformed

    def get_projection_matrix(self) -> np.array:
        """Return 4x4 projection matrix."""
        return self.projector

    @staticmethod
    def get_translation_matrix(T_x: float, T_y: float, T_z: float) -> np.array:
        """Return 4x4 translation matrix as 2D np.array."""
        return np.array([[1.0,  0.0,  0.0,  T_x],
                         [0.0,  1.0,  0.0,  T_y],
                         [0.0,  0.0,  1.0,  T_z],
                         [0.0,  0.0,  0.0,  1.0]])

    @staticmethod
    def get_X_rotation_matrix(degree: float) -> np.array:
        """Return 4x4 rotation by X matrix by angle given in degrees as 2D np.array."""
        phi = np.deg2rad(degree)
        return np.array([[1.0,      0.0,           0.0,      0.0],
                         [0.0,  np.cos(phi),  -np.sin(phi),  0.0],
                         [0.0,  np.sin(phi),   np.cos(phi),  0.0],
                         [0.0,      0.0,           0.0,      1.0]])

    @staticmethod
    def get_Y_rotation_matrix(degree: float) -> np.array:
        """Return 4x4 rotation by Y matrix by angle given in degrees as 2D np.array."""
        phi = np.deg2rad(degree)
        return np.array([[ np.cos(phi),  0.0,  np.sin(phi),  0.0],
                         [     0.0,      1.0,      0.0,      0.0],
                         [-np.sin(phi),  0.0,  np.cos(phi),  0.0],
                         [     0.0,      0.0,      0.0,      1.0]])

    @staticmethod
    def get_Z_rotation_matrix(degree: float) -> np.array:
        """Return 4x4 rotation by Z matrix by angle given in degrees as 2D np.array."""
        phi = np.deg2rad(degree)
        return np.array([[np.cos(phi),  -np.sin(phi),  0.0,  0.0],
                         [np.sin(phi),   np.cos(phi),  0.0,  0.0],
                         [    0.0,           0.0,      1.0,  0.0],
                         [    0.0,           0.0,      0.0,  1.0]])

    @staticmethod
    def get_scale_matrix(S_x: float, S_y: float, S_z: float) -> np.array:
        """Return 4x4 scaling matrix as 2D np.array."""
        return np.array([[S_x,  0.0,  0.0,  0.0],
                         [0.0,  S_y,  0.0,  0.0],
                         [0.0,  0.0,  S_z,  0.0],
                         [0.0,  0.0,  0.0,  1.0]])

    @staticmethod
    def vector_from_3d_to_homo(vector: np.array, w_value=1) -> np.array:
        """
        Cast 3D vector to the homogeneous coordinates
        with the fourth equal to w_value with 1 by default.
        Return 4x1 vector as 1D np.array.
        """
        return np.array([vector[0], vector[1], vector[2], w_value], dtype=object)

    @staticmethod
    def vector_from_homo_to_3d(vector: np.array) -> np.array:
        """
        Cast the vector in homogeneous coordinates to the 3D one
        by normalizing each coordinate X, Y and Z by the fourth entry.
        Return 3x1 vector as 1D np.array.
        """
        w = vector[3]
        if w == 0: w = 1
        return np.array([float(vector[0] / w), float(vector[1] / w), float(vector[2] / w)])

    @staticmethod
    def matrix_4x4_mul_vector_4x1(matrix_4x4: np.array, vector_3x1: np.array) -> np.array:
        """
        Cast 3D vector to the homogeneous coordinates and multiply it from left
        by the 4x4 matrix. Return 4x1 vector in homogeneous coordinates as 1D np.array.
        """
        return np.matmul(matrix_4x4, Engine.vector_from_3d_to_homo(vector_3x1))

    @staticmethod
    def get_normal(triangle: np.array) -> np.array:
        """
        Get the normal vector of a given triangle.
        Return its normalized form as a 1D np.array.
        """
        edge_1 = triangle[1] - triangle[0]
        edge_2 = triangle[2] - triangle[0]

        normal = np.cross(edge_1, edge_2)

        return normal / np.linalg.norm(normal)


if __name__ == "__main__":
    demo = Engine("3d_demo", (500, 500), (100, 100), 10.0, (0, 102, 200), "data/cube.obj")
    # TODO: point_size argument seems to be useless
    demo.start()
