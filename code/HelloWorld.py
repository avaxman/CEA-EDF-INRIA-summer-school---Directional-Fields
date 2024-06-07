import os
import polyscope as ps
import polyscope.imgui as psim
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import spsolve
import cmath



def load_off_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Parse the vertices and faces from the OFF file
    num_vertices, num_faces, _ = map(int, lines[1].split())

    vertices = np.array([list(map(float, line.split())) for line in lines[2:2 + num_vertices]])
    faces = np.array([list(map(int, line.split()))[1:] for line in lines[2 + num_vertices:]])

    return vertices, faces




if __name__ == '__main__':
    ps.init()

    vertices, faces = load_off_file(os.path.join('..', 'data', 'fandisk.off'))

    ps_mesh = ps.register_surface_mesh("Hello World Mesh", vertices, faces)

    ps.show()
