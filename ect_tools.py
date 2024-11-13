import numpy as np
from numpy import cross, eye
from scipy.linalg import expm, norm

def fast_norm(a):
    n = fast_dot(a, a) ** 0.5
    return n

def fast_dot(a, b):
    d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    return d

def fast_cross(a, b):
    c = np.empty(3)
    c[0] = a[1] * b[2] - a[2] * b[1]
    c[1] = a[2] * b[0] - a[0] * b[2]
    c[2] = a[0] * b[1] - a[1] * b[0]
    return c

def sort_polygon(points):
    n = points[0] / fast_norm(points[0])
    proj_pts = [points[i] - fast_dot(points[i], n) * n for i in range(len(points))]
    angles = [np.sign(fast_dot(n, fast_cross(proj_pts[1], proj_pts[i + 1]))) * np.degrees(np.arccos(
        np.clip(fast_dot(proj_pts[1] / fast_norm(proj_pts[1]), proj_pts[i + 1] / fast_norm(proj_pts[i + 1])), -1, 1)))
              for i in range(len(points) - 1)]
    angles = np.array(angles)
    sorted_indices = angles.argsort()
    indices = [0, *sorted_indices + 1]
    return indices


def rotation_matrix_about_axis(axis, theta):
    return expm(cross(eye(3), axis / norm(axis) * theta))

def normalize(vector):
    return vector / np.sum(vector ** 2) ** 0.5


def triangulate_a_tomato(triangle, index=0):
    """
    A function for triangulating tomato-wedge triangles
    (a triangle that essentially contains two half great circles)
    triangle: the triangle in question, (p0,p1,p2)
    index: which entry of the triangle is the basis (0,1 or 2)

    Returns: triangulation of the directions where index is lower than the rest
    This region looks like a tomato wedge: the ends are the main_dir, and the midpoints of the edges are
    t1 and t2. These are obtained by rotating the ends along the axes defined by the cuts
    t1: Point where p0=p1 and p0 has negative sign
    t2: Point where p0=p2 and p0 has negative sign
    """
    p0 = triangle[index]
    p1 = np.delete(triangle, index, 0)[0]
    p2 = np.delete(triangle, index, 0)[1]
    main_dir = normalize(np.cross(p1 - p0, p2 - p0))  # The triple points
    # Next we rotate the triple point about p0=p1, p0=p2
    theta = np.pi / 2  # Rotate half a circle
    axis1 = normalize(p1 - p0)
    axis2 = normalize(p2 - p0)
    t1 = np.dot(rotation_matrix_about_axis(axis1, theta), main_dir)
    t2 = np.dot(rotation_matrix_about_axis(axis2, theta), main_dir)

    if np.dot(t1, p0 - p2) > 0:
        t1 = -t1
    if np.dot(t2, p0 - p1) > 0:
        t2 = -t2
    stack = np.stack([main_dir, -main_dir, t1, t2])
    triangles = np.stack([[0, 2, 3], [1, 2, 3]])
    return stack, triangles


def unique_list(a_list: list) -> list:
    """
    A helper function for unique elements of a list of non-hashable elements.
    Args:
         a_list:
             a list
    Returns
        uniques:
            a list of unique entries of the list a_list
    """
    uniques = []
    used = set()
    for item in a_list:
        tmp = repr(item)
        if tmp not in used:
            used.add(tmp)
            uniques.append(item)
    return uniques
