import math
import numpy as np
from scipy import integrate
from scipy.stats import special_ortho_group
from ect_tools import fast_norm, fast_dot, fast_cross
tol = 0.000001


def rotation_check(polygon, point):
    # Here p1, p2, p3, pi are the Cartesian coordinates
    # P spherical polygon
    # pi not necessarily on sphere
    n = len(polygon)

    # Detect poles
    for i in range(n):
        if np.abs(polygon[i][2]) > 1 - tol: return False

    if point[0] == 0 and point[1] == 0: return False

    # Detect phi=0 arc
    for i in range(n):
        if polygon[i][0] > 0 and np.abs(polygon[i][1]) < tol: return False

    if point[0] > 0 and np.abs(point[1]) < tol: return False

    # Detect meridian
    for i in range(n):
        if np.abs(polygon[i][0] * polygon[(i + 1) % n][1] - polygon[i][1] * polygon[(i + 1) % n][0]) < tol * np.sum(
                (polygon[i] - polygon[(i + 1) % n]) ** 2):
            return False

    # Detect equator
    for i in range(n):
        if polygon[i][2] == 0 and polygon[(i + 1) % n][2] == 0: return False

    # Detect crossing arc
    phis = []
    for i in range(n):
        phi = math.atan2(polygon[i][1], polygon[i][0])
        if phi < 0: phi += 2 * np.pi
        phis.append(phi)
    for i in range(n):
        if np.abs(phis[i] - phis[(i + 1) % n]) > np.pi: return False
    return True


def roted_sphere(polygon, point):
    rotation_check_value = rotation_check(polygon, point)
    while not rotation_check_value:
        rotation_matrix = special_ortho_group.rvs(3)
        for i in range(len(polygon)):
            polygon[i] = np.dot(rotation_matrix, polygon[i])
        point = np.dot(rotation_matrix, point)
        rotation_check_value = rotation_check(polygon, point)
    return polygon, point


def cartesian_to_spherical(x, y, z):
    tau = math.asin(z)
    phi = math.atan2(y, x)
    if math.atan2(y, x) < 0:
        phi += 2 * np.pi
    return phi, tau


def solve_great_circle(phi_1, tau_1, phi_2, tau_2):
    # compute phi_0, a
    r1 = np.cos(phi_1) * np.tan(tau_2) - np.cos(phi_2) * np.tan(tau_1)
    r2 = -np.sin(phi_1) * np.tan(tau_2) + np.sin(phi_2) * np.tan(tau_1)

    if np.abs(r2) < tol:
        phi_0 = np.pi / 2
        # phi_0 can be -pi/2 with 'a' becoming '-a', we get the same equation                                
    else:
        phi_0 = np.arctan(r1 / r2)

    # cos = 0, use another point
    if np.abs(np.abs((phi_1 - phi_0) % np.pi) - np.pi / 2) < tol:
        a = np.tan(tau_2) / np.cos(phi_2 - phi_0)
    else:
        a = np.tan(tau_1) / np.cos(phi_1 - phi_0)

    return phi_0, a


def integrate_arc(phi_1, tau_1, phi_2, tau_2, phi_i, tau_i):
    phi_0, a = solve_great_circle(phi_1, tau_1, phi_2, tau_2)

    def f1(phi):
        return (1 - np.power(a, 2) * np.power(np.cos(phi - phi_0), 2)) / (
                    1 + np.power(a, 2) * np.power(np.cos(phi - phi_0), 2))

    def f2(phi):
        return (2 * a * np.cos(phi - phi_0) * np.cos(phi - phi_i)) / (
                    1 + np.power(a, 2) * np.power(np.cos(phi - phi_0), 2))

    def f3(phi):
        return np.arctan(a * np.cos(phi - phi_0)) * np.cos(phi - phi_i)

    I1, error1 = integrate.quad(f1, phi_1, phi_2)
    I2, error2 = integrate.quad(f2, phi_1, phi_2)
    I3, error3 = integrate.quad(f3, phi_1, phi_2)

    integral = 0.25 * np.sin(tau_i) * I1 - 0.25 * np.cos(tau_i) * I2 - 0.5 * np.cos(tau_i) * I3
    return integral


def integrate_polygon(polygon, point):
    integral = 0
    n = len(polygon)
    polygon, point = roted_sphere(polygon, point)
    r = fast_norm(point)

    phi_i, tau_i = cartesian_to_spherical(point[0] / r, point[1] / r, point[2] / r)
    phis = []
    taus = []

    for i in range(n):
        phi, tau = cartesian_to_spherical(polygon[i][0], polygon[i][1], polygon[i][2])
        phis.append(phi)
        taus.append(tau)

    for i in range(n):
        integral += integrate_arc(phis[i], taus[i], phis[(i + 1) % n], taus[(i + 1) % n], phi_i, tau_i)

    integral = r * integral

    return integral


def fast_compare(a, b):
    if np.abs(a[0] - b[0]) < tol and np.abs(a[1] - b[1]) < tol and np.abs(a[2] - b[2]) < tol:
        return True
    return False


def fast_clip(r):
    if r > 1:
        r = 1
    elif r < -1:
        r = -1
    return r


def spherical_angle(p1, p2, p3):
    # Check coinciding point/points on the same arc to avoid /0
    # then v1_raw, v2_raw can't be 0 in this case
    v1_raw = fast_cross(fast_cross(p2, p1), p2)
    v2_raw = fast_cross(fast_cross(p2, p3), p2)

    # Special case of antipodal points
    if fast_norm(v1_raw) == 0 or fast_norm(v2_raw) == 0:
        return None

    v1 = v1_raw / fast_norm(v1_raw)
    v2 = v2_raw / fast_norm(v2_raw)

    inprod = fast_dot(v1, v2)
    inprod = fast_clip(inprod)
    return np.arccos(inprod)


def polygon_area(polygon):
    n = len(polygon)
    polygon_angles = 0
    for i in range(n):
        polygon[i] = polygon[i] / fast_norm(polygon[i])
    for i in range(n):
        if spherical_angle(polygon[i], polygon[(i + 1) % n], polygon[(i + 2) % n]) is None: return None
        polygon_angles += spherical_angle(polygon[i], polygon[(i + 1) % n], polygon[(i + 2) % n])
    area = polygon_angles - (n - 2) * np.pi
    if area <= -tol: print('error in polygon_area')
    return polygon_angles - (n - 2) * np.pi


def update_polygon(P, N, n):
    l = len(P)
    Tol_upp = 0.000000000000001  # e-15
    Tol_bot = 0.00000001  # e-8
    value = []
    A = []
    P_updated = []
    N_updated = []
    l_approx_pos = 0

    for i in range(l):
        a = fast_dot(P[i], n)
        A.append(a)
        if a > -Tol_upp:
            l_approx_pos += 1
            if a >= 0: value.append(1)
        if a < 0: value.append(-1)
    # Include one collinearity
    if l_approx_pos == l: return P, N
    if l_approx_pos == 0: return None, None
    # Test another collinearity
    for i, a in enumerate(A):
        current_in_upp_range = -Tol_upp < A[i] < Tol_upp
        next_in_bot_range = -Tol_bot < A[(i + 1) % l] < Tol_bot
        current_in_bot_range = -Tol_bot < A[i] < Tol_bot
        next_in_upp_range = -Tol_upp < A[(i + 1) % l] < Tol_upp
        condition1 = current_in_upp_range and next_in_bot_range
        condition2 = current_in_bot_range and next_in_upp_range
        if condition1 or condition2:
            if A[i - 1] > 0:
                return P, N
            elif A[i - 1] < 0:
                return None, None
            else:
                print('wrong A in update_polygon', A)

    n_update = 2
    for i in range(l):
        if value[i] == 1:
            P_updated.append(P[i])
            N_updated.append(N[i])
        if value[i] * value[(i + 1) % l] == -1:
            p_raw = fast_cross(N[i], n)
            if value[i] == -1: p_raw = -p_raw
            p = p_raw / fast_norm(p_raw)
            P_updated.append(p)
            if n_update > 0:
                n_update -= 1
                if value[i] == 1:
                    N_updated.append(n)
                else:
                    N_updated.append(N[i])

    P_updated, N_updated = P_simplify(P_updated, N_updated)
    return P_updated, N_updated


def P_simplify(P_updated, N_updated):
    '''
    In order to fix cases that arcs go through points.
    With too short arc, i.e. too closed points, cross product is probably inaccurate.
    '''
    min_len = 0.0000004  # e-7
    P_tmp = []
    N_tmp = []
    P_ind = []
    l_update = len(P_updated)
    for i in range(l_update):
        vec_edge_sq = (P_updated[i] - P_updated[(i + 1) % l_update]) ** 2
        l_edge = (vec_edge_sq[0] + vec_edge_sq[1] + vec_edge_sq[2]) ** 0.5
        # correct edge normal
        if l_edge < min_len: continue
        P_ind.append(i)
    if len(P_ind) <= 2:
        return None, None
    else:
        l_ind = len(P_ind)
        for i, ind in enumerate(P_ind):
            P_tmp.append(P_updated[ind])
            if (ind + 1) % l_update == P_ind[(i + 1) % l_ind]:
                N_tmp.append(N_updated[ind])
            else:
                n_tmp = fast_cross(P_updated[ind], P_updated[P_ind[(i + 1) % l_ind]])
                n = n_tmp / fast_norm(n_tmp)
                N_tmp.append(n)
    return np.array(P_tmp), np.array(N_tmp)


def P_intersect(P1, N1, N2):
    '''
    P=[v_1, v_2, ..., v_n] is one polygon (use smaller one, less computation before possible drop)
    N=[n_1, n_2, ..., n_m] is the edge normals of polygon
    P, N: numpy array
    '''
    P_updated = np.array(P1)
    N1_updated = np.array(N1)
    for n in N2:
        P_updated, N1_updated = update_polygon(P_updated, N1_updated, n)
        if P_updated is None: return None, None
    return P_updated, N1_updated


def P_div(Pi, Pj, Ni, Nj, pi, pj):
    if len(Pi) <= len(Pj):
        P_int, N_int = P_intersect(Pi, Ni, Nj)
    else:
        P_int, N_int = P_intersect(Pj, Nj, Ni)

    # nothing
    if P_int is None:
        return None, None, None

    # identical
    if (np.abs(pi - pj) < tol).all():
        return P_int, None, polygon_area(P_int)

    P_divi, _ = update_polygon(P_int, N_int, pi - pj)
    P_divj, _ = update_polygon(P_int, N_int, pj - pi)

    return P_divi, P_divj, polygon_area(P_int)


def ECT_distance_s(ECT):
    '''
    d_ECT = \int\sum(ECT1-ECT2)^2 
          = \int\sum (ECT1[i]*ECT1[j] + ECT2[i]*ECT2[j] - 2 * ECT1[i] * ECT2[j])
    ECT_s computes the first two terms
    ECT_d computes the last term
    '''
    integral = 0

    for idx1 in range(len(ECT)):
        for idx2 in range(len(ECT)):
            integral_i = 0
            integral_j = 0
            integral_s = 0
            if ECT[idx1][3] == ECT[idx2][3] and idx1 == idx2:
                P = np.array(ECT[idx1][2])
                integral_i = integrate_polygon(P, ECT[idx1][1])
                integral_s = polygon_area(P)
            elif ECT[idx1][3] == ECT[idx2][3] and idx1 != idx2:
                continue
            else:
                P_divi, P_divj, integral_s = P_div(ECT[idx1][2], ECT[idx2][2], ECT[idx1][4], ECT[idx2][4], ECT[idx1][1],
                                                   ECT[idx2][1])
                if integral_s is not None:
                    if P_divi is not None:
                        integral_i = integrate_polygon(P_divi, ECT[idx1][1])
                    if P_divj is not None:
                        integral_j = integrate_polygon(P_divj, ECT[idx2][1])
                else:
                    continue
            integral += ECT[idx1][0] * ECT[idx2][0] * (integral_s - integral_i - integral_j)

    return integral


def ECT_distance_d(ECT1, ECT2):
    integral = 0

    for idx1 in range(len(ECT1)):
        for idx2 in range(len(ECT2)):
            integral_i = 0
            integral_j = 0
            integral_s = 0
            P_divi, P_divj, integral_s = P_div(ECT1[idx1][2], ECT2[idx2][2], ECT1[idx1][4], ECT2[idx2][4],
                                               ECT1[idx1][1], ECT2[idx2][1])
            if integral_s is not None:
                if P_divi is not None:
                    integral_i = integrate_polygon(P_divi, ECT1[idx1][1])
                if P_divj is not None:
                    integral_j = integrate_polygon(P_divj, ECT2[idx2][1])
            else:
                continue
            integral += ECT1[idx1][0] * ECT2[idx2][0] * (integral_s - integral_i - integral_j)

    return integral


############# Auxiliary Functions ##############
def poly_orientation(poly):
    n = fast_cross(poly[0], poly[1])
    if fast_dot(n, poly[2]) < 0:
        tmp = poly.copy()
        reversed_poly = tmp[::-1]
        poly = reversed_poly
    M = len(poly)
    for i in range(M):
        tmp1 = poly[i].copy()
        normed_poly = tmp1 / fast_norm(tmp1)
        poly[i] = normed_poly
    return poly


def return_ECT(s1):
    tmp = []
    for key in s1.clean_polygon_gains:
        # TMP: gain for each vertices
        TMP = s1.clean_polygon_gains[key]
        for j in range(TMP.shape[0]):
            megatmp = []
            megatmp.append(TMP[j])
            megatmp.append(s1.V[key, :])
            poly = s1.polygon_angles[key][s1.clean_polygons[key][j]]
            if len(poly) < 3: continue
            poly = poly_orientation(poly)
            megatmp.append(poly)
            megatmp.append(key)
            N = compute_normal(poly)
            megatmp.append(N)
            tmp.append(megatmp)
    return tmp


def compute_normal(polygon):
    # Given a polygon, return its edge normals
    l = len(polygon)
    normals = []
    for i in range(l):
        raw_normal = fast_cross(polygon[i], polygon[(i + 1) % l])
        normal = raw_normal / fast_norm(raw_normal)
        normals.append(normal)
    normals = np.array(normals)
    return normals
