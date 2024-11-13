import itertools
import math
import numpy as np
import ect_tools


class Shape:
    # params: self, Vertices, Edges, Triangles
    def __init__(self, vertices, triangles, name=None, prepare=False):
        self.V = vertices
        self.T = triangles
        self.name = name
        self.links = {}
        self.polygon_angles = {}
        self.polygon_midpoints = {}
        self.vertex_faces = {}
        self.vertex_edges = {}
        self.polygon_gains = {}
        self.clean_polygon_gains = {}
        self.polygons = {}
        self.clean_polygon_gains = {}
        self.clean_polygons = {}
        self.edges = np.zeros([0, 2])
        self.triangles = np.zeros([0, 3])
        if prepare:
            self.prepare()
    def prepare(self):
        """
        Performs the overhead computations necessary for the ECT
        """
        self.center_n_scale()
        self.prepare_for_ECT()
        self.compute_links()
        self.compute_polygons()
        self.compute_gains()
        self.clean_gains()
    def center_n_scale(self):
        self.V = self.V - np.mean(self.V, 0)
        scales = [sum(tmp ** 2) ** 0.5 for tmp in self.V]
        self.V = self.V / max(scales)

    def compute_links(self):
        """
        Links for each vertex, 1st step of the algorithm
        """
        # Go through the triangles and get all neighbors for all vertices
        for i in range(len(self.V)):
            a = np.where(self.T[:, 0] == i)
            b = np.where(self.T[:, 1] == i)
            c = np.where(self.T[:, 2] == i)
            A = np.delete(self.T[a, :], 0, axis=2)
            B = np.delete(self.T[b, :], 1, axis=2)
            C = np.delete(self.T[c, :], 2, axis=2)
            neighbor_array = np.concatenate((A, B, C), 1)
            self.links[i] = np.unique(neighbor_array)

    def compute_lower_stars(self, triplet):
        triangle = self.V[triplet, :]
        direction = self.compute_face_normal(triangle)
        key = triplet[0]
        nbd = [key, *self.links[key].astype(int)]
        tmp2 = np.matmul(direction.reshape(1, -1), self.V[nbd].T)
        tmp3 = tmp2 - tmp2[0][0]
        ls_inds = np.where(tmp3 <= 0)[1]
        ls = [nbd[i] for i in ls_inds]
        set0 = set.difference(set(ls), set(triplet))
        set1 = set.union(set0, {triplet[0]})
        set2 = set.union(set1, set.union({triplet[0]}, {triplet[1]}))
        set3 = set.union(set1, set.union({triplet[0]}, {triplet[2]}))
        set4 = set.union(set1, set(triplet))
        r1 = [set1, set2, set3, set4]
        direction = -direction
        tmp2 = np.matmul(direction.reshape(1, -1), self.V[nbd].T)
        tmp3 = tmp2 - tmp2[0][0]
        ls_inds = np.where(tmp3 <= 0)[1]
        ls = [nbd[i] for i in ls_inds]
        set0 = set.difference(set(ls), set(triplet))
        set1 = set.union(set0, {triplet[0]})
        set2 = set.union(set1, set.union({triplet[0]}, {triplet[1]}))
        set3 = set.union(set1, set.union({triplet[0]}, {triplet[2]}))
        set4 = set.union(set1, set(triplet))
        r2 = [set1, set2, set3, set4]
        return r1, r2

    def prepare_for_discrete_ECT(self):
        """
        A helper function for computing discrete Euler curves.
        Non-digital function
        """
        tmp = np.zeros([0, 2])
        for key in self.vertex_edges:
            array = self.vertex_edges[key]
            tmp = np.concatenate([tmp, array])
        edges = np.unique(tmp, axis=0)
        self.edges = edges.astype(int)
        self.triangles = self.T.astype(int)

    def compute_discrete_ECT(self, direction, threshold):
        """
        Evaluates the ECT at a given direction and height
        Non-digital function
        """
        heights = np.matmul(direction, self.V.T)
        edge_heights = np.max(heights[self.edges], 1)
        triangle_heights = np.max(heights[self.triangles], 1)
        summa = sum(heights <= threshold) - sum(edge_heights <= threshold) + sum(triangle_heights <= threshold)
        return summa

    def compute_polygon(self, key):
        trios = list()
        lowerstars = list()
        neighbors = self.links[key]
        comparisons = np.array([*neighbors]).astype(int)
        points = np.zeros([2 * math.comb(len(comparisons), 2), 3])
        duos = itertools.combinations(comparisons, 2)
        i = 0
        for duo in duos:
            trio = np.array([key, *duo])
            trios.append(trio)
            trios.append(trio)
            triangle = self.V[trio, :]
            n1 = self.compute_face_normal(triangle)
            l1, l2 = self.compute_lower_stars(trio)
            lowerstars.append(l1)
            points[i, :] = n1
            i = i + 1
            points[i, :] = -n1
            lowerstars.append(l2)
            i = i + 1
        return trios, points, lowerstars

    def compute_polygons(self):
        for key in self.links:
            neighbors=self.links[key]
            polygons=list()
            midpoints=list() #polygon midpoints
            if(len(neighbors)==2): # The case of an isolated triangle
                triangleind=np.array([key,*neighbors]).astype(int)
                triangle=self.V[triangleind,:]
                V,T=ect_tools.triangulate_a_tomato(triangle)
                for i in range(T.shape[0]):
                    tmp=np.mean(V[T[i]],0)
                    midpoints.append(tmp/(sum(tmp**2))**(0.5))
                    polygons.append(T[i])
                self.polygon_angles[key]=V
                self.polygons[key]=polygons
                self.polygon_midpoints[key]=midpoints
                continue
            trios,points,lowerstars=(self.compute_polygon(key))
            allstars=list(itertools.chain.from_iterable(lowerstars))
            stars=list(set(tuple(sorted(s)) for s in allstars))
            stardiaries=list()
            for star in stars:
                st=set(star)
                stardiary=list()
                for i in range(len(lowerstars)):
                    tmptrick=[tmp==st for tmp in lowerstars[i]]
                    if(sum(tmptrick)==1):
                        stardiary.append(i)
                stardiaries.append(stardiary)
            for stardiary in stardiaries:
                pts=points[stardiary]
                inds=ect_tools.sort_polygon(pts)
                polygon=[stardiary[ind] for ind in inds]
                polygons.append(polygon)
                tmp=np.mean(points[polygon,:],0)
                midpoint=tmp/(sum(tmp**2))**(0.5)
                midpoints.append(midpoint)
            self.polygons[key]=polygons
            self.polygon_angles[key]=points
            self.polygon_midpoints[key]=midpoints

    def compute_face_normal(self, triangle):
        x1 = triangle[0, :]
        x2 = triangle[1, :]
        x3 = triangle[2, :]
        tmp = np.cross(x2 - x1, x3 - x1)
        return tmp / sum(tmp ** 2) ** 0.5

    def compute_gains(self):
        """
        Fourth step: evaluate the ECT gains in each triangle of the Delaunay triangulation.
        """
        for key in self.polygon_midpoints:
            directions = self.polygon_midpoints[key]
            gains = np.zeros([len(directions), 1])
            for i in range(len(directions)):
                direction = directions[i]
                gains[i] = self.evaluate_local_ECT(direction, key)
            self.polygon_gains[key] = gains
        return True

    def clean_gains(self):
        """
        Fourth step: evaluate the ECT gains in each triangle of the Delaunay triangulation.
        """
        for key in self.polygon_midpoints:
            gains = self.polygon_gains[key]
            polygons = self.polygons[key]
            inds = np.where(gains != 0)[0]
            clean_polygons = [polygons[ind] for ind in inds]
            clean_gains = gains[inds]
            self.clean_polygon_gains[key] = clean_gains
            self.clean_polygons[key] = clean_polygons
        return True

    def evaluate_local_ECT(self, direction, vertex):
        """
        Helper function for finding the spherical triangles that matter.
        This is the combinatorial algorithm
        """
        dir_vector = direction
        heights = np.matmul(dir_vector, self.V.T)
        order = np.argsort(heights)
        cutpoint = np.where(order == vertex)[0][0]
        subset1 = order[:cutpoint]
        subset2 = order[:(cutpoint + 1)]
        faces = self.vertex_faces[vertex]
        edges = self.vertex_edges[vertex]
        # Change in ECT: 1 (p0) -edges + triangles
        chi = 1 + sum([set(subset2).issuperset(row) for row in faces.tolist()]) \
              - sum([set(subset1).issuperset(row) for row in faces.tolist()]) \
              - sum([set(subset2).issuperset(row) for row in edges.tolist()]) \
              + sum([set(subset1).issuperset(row) for row in edges.tolist()])
        return (chi)

    def prepare_for_ECT(self):
        """
        This helper function performs overhead computations to
        quicken the combinatorial ECT evaluation algorithm
        """
        for i in range(len(self.V)):
            a = self.T[np.where(self.T[:, 0] == i), :]
            b = self.T[np.where(self.T[:, 1] == i), :]
            c = self.T[np.where(self.T[:, 2] == i), :]
            faces = np.concatenate((a, b, c), 1)[0]
            # Initialize edges: At most 3 * faces
            edges = np.zeros((3 * faces.shape[0], 2))
            for j in range(faces.shape[0]):
                v0 = min(faces[j, (0, 1)])
                v1 = max(faces[j, (0, 1)])
                v2 = min(faces[j, (0, 2)])
                v3 = max(faces[j, (0, 2)])
                v4 = min(faces[j, (1, 2)])
                v5 = max(faces[j, (1, 2)])
                edges[3 * j, :] = [v0, v1]
                edges[3 * j + 1, :] = [v2, v3]
                edges[3 * j + 2, :] = [v4, v5]
            edges = np.unique(edges, axis=0)
            ind1 = edges[:, 0] == i
            ind2 = edges[:, 1] == i
            edges = edges[np.where(np.any([ind1, ind2], 0)), :][0]
            verts = np.unique(edges, 0)
            self.vertex_faces[i] = faces
            self.vertex_edges[i] = edges
