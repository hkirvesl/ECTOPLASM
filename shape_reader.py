import numpy as np
from numpy import empty
from shape import Shape

# Currently supports only triangulated meshes. So no extra edges
# Also: The algorithm doesn't allow just yet the vertices that belong to just 1 triangle
class ShapeReader(object):
    @staticmethod
    def shape_from_file(file_path, prepare=False):
        namelist = file_path.split('/')
        name = namelist[len(namelist) - 1].split('.')[0]
        vertices, faces = ShapeReader.off_parser(file_path)
        return Shape(vertices=vertices, triangles=faces, name=name, prepare=prepare)

    @staticmethod
    def off_parser(file_path):
        file = open(file_path, "r")
        # Checking we have valid headers
        header = file.readline().split()
        if header[0] != 'OFF':
            msg = 'The input file does not seem to be valid off file, first line should read "OFF".'
            raise TypeError(msg)
        # Reading in the number of vertices, faces and edges, and pre-formatting their arrays
        (v, f, e) = map(int, file.readline().strip().split(' '))
        vertices = empty([v, 3], dtype=np.float32)
        faces = empty([f, 3])
        # Read in the vertices
        for i in range(0, v):
            vertices[i] = list(map(float, file.readline().strip().split(' ')))
        # Read in the faces
        for i in range(0, f):
            line = list(map(int, file.readline().strip().split(' ')))
            # Notify the user that there are non-triangular faces.
            # Non-triangular faces wouldn't be supported by the vtk setup that we have anyway.
            # Better way would be to triangulate the polygons, that can be added if deemed useful
            # Also, we could use warnings
            if len(line) != 4:
                print("Warning: The .off contains non-triangular faces, holes might have been created.")
            if line[0] != 3 and len(line) == 4:
                print(
                    "Warning: The .off file contains a face that is defined to be non-triangular. It is a valid triangle, reading it as a triangle.")
            faces[i] = line[1:4]
        vertices.astype(np.float32)
        faces.astype(int)
        return vertices, faces
