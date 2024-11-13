import numpy as np
import torch
from scipy.stats import special_ortho_group
from torchquad import Trapezoid
tp = Trapezoid()
tol = torch.tensor(0.000001)


def rotation_check(P, pi):
    # Here p1, p2, p3, pi are the Cartesian coordinates
    # P sperical polygon
    # pi not necessarily on sphere
    N = len(P)

    # Detect poles
    for i in range(N):
        if torch.abs(P[i][2]) > 1 - tol: return False

    if pi[0] == 0 and pi[1] == 0: return False

    # Detect phi=0 arc
    for i in range(N):
        if P[i][0] > 0 and torch.abs(P[i][1]) < tol: return False

    if pi[0] > 0 and torch.abs(pi[1]) < tol: return False

    # Detect meridian
    for i in range(N):
        # if np.abs(np.abs(P[i][0]/P[i][1])-np.abs(P[(i+1)%N][0]/P[(i+1)%N][1])) < tol: return False
        if torch.abs(P[i][0] * P[(i + 1) % N][1] - P[i][1] * P[(i + 1) % N][0]) < tol * torch.sum(
                (P[i] - P[(i + 1) % N]) ** 2):
            return False

    # Detect equator
    for i in range(N):
        if P[i][2] == 0 and P[(i + 1) % N][2] == 0: return False

    # Detect crossing arc
    phis = []
    for i in range(N):
        phi = torch.atan2(P[i][1], P[i][0])
        if phi < 0: phi += 2 * torch.pi
        phis.append(phi)
    for i in range(N):
        if torch.abs(phis[i] - phis[(i + 1) % N]) > torch.pi: return False
    return True


def roted_sphere(P, pi):
    truth_value = rotation_check(P, pi)
    while truth_value == False:
        rotation_matrix = torch.tensor(special_ortho_group.rvs(3))
        for i in range(len(P)):
            P[i] = torch.matmul(rotation_matrix, P[i])
        pi = torch.matmul(rotation_matrix, pi.double())
        truth_value = rotation_check(P, pi)
    return P, pi


def cartesian_to_spherical(x, y, z):
    tau = torch.asin(z)
    phi = torch.atan2(y, x)
    if torch.atan2(y, x) < 0:
        phi += 2 * torch.pi
    return phi, tau


def solve_great_circle(phi_1, tau_1, phi_2, tau_2):
    # compute phi_0, a
    r1 = torch.cos(phi_1) * torch.tan(tau_2) - torch.cos(phi_2) * torch.tan(tau_1)
    r2 = -torch.sin(phi_1) * torch.tan(tau_2) + torch.sin(phi_2) * torch.tan(tau_1)

    if torch.abs(r2) < tol:
        phi_0 = torch.tensor(torch.pi / 2)
        # phi_0 can be -pi/2 with 'a' becoming '-a', we get the same equation                                
    else:
        phi_0 = torch.arctan(r1 / r2)

    # cos = 0, use another point
    if torch.abs(torch.abs((phi_1 - phi_0) % torch.pi) - torch.pi / 2) < tol:
        a = torch.tan(tau_2) / torch.cos(phi_2 - phi_0)
    else:
        a = torch.tan(tau_1) / torch.cos(phi_1 - phi_0)

    return phi_0, a


def int_arc(phi_1, tau_1, phi_2, tau_2, phi_i, tau_i):
    phi_0, a = solve_great_circle(phi_1, tau_1, phi_2, tau_2)

    # print(phi_0)
    # print(a)
    # print('integration')
    def f1(phi):
        return (1 - torch.pow(a, 2) * torch.pow(torch.cos(phi - phi_0), 2)) / (
                    1 + torch.pow(a, 2) * torch.pow(torch.cos(phi - phi_0), 2))

    def f2(phi):
        return (2 * a * torch.cos(phi - phi_0) * torch.cos(phi - phi_i)) / (
                    1 + torch.pow(a, 2) * torch.pow(torch.cos(phi - phi_0), 2))

    def f3(phi):
        return torch.arctan(a * torch.cos(phi - phi_0)) * torch.cos(phi - phi_i)

    I1 = tp.integrate(f1, dim=1, integration_domain=[[phi_1, phi_2]])
    I2 = tp.integrate(f2, dim=1, integration_domain=[[phi_1, phi_2]])
    I3 = tp.integrate(f3, dim=1, integration_domain=[[phi_1, phi_2]])

    integral = 0.25 * torch.sin(tau_i) * I1 - 0.25 * torch.cos(tau_i) * I2 - 0.5 * torch.cos(tau_i) * I3
    return integral


def int_polygon(P, pi):
    I = torch.tensor(0, dtype=torch.float64)
    N = len(P)
    P_tmp, pi_tmp = roted_sphere(P.clone(), pi)
    r = fast_norm(pi)

    phi_i, tau_i = cartesian_to_spherical(pi_tmp[0] / r, pi_tmp[1] / r, pi_tmp[2] / r)
    phis = []
    taus = []

    for i in range(N):
        phi, tau = cartesian_to_spherical(P_tmp[i][0], P_tmp[i][1], P_tmp[i][2])
        phis.append(phi)
        taus.append(tau)

    for i in range(N):
        I += int_arc(phis[i], taus[i], phis[(i + 1) % N], taus[(i + 1) % N], phi_i, tau_i)

    I = r * I

    return I


def fast_cross(a, b):
    c = torch.empty(3)
    c[0] = a[1] * b[2] - a[2] * b[1]
    c[1] = a[2] * b[0] - a[0] * b[2]
    c[2] = a[0] * b[1] - a[1] * b[0]
    return c


def fast_dot(a, b):
    d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    return d


def fast_norm(a):
    n = fast_dot(a, a) ** 0.5
    return n


def fast_compare(a, b):
    if torch.abs(a[0] - b[0]) < tol and torch.abs(a[1] - b[1]) < tol and torch.abs(a[2] - b[2]) < tol:
        return True
    return False


def fast_clip(r):
    if r > 1:
        r = 1
    elif r < -1:
        r = -1
    return r


def sph_angle(p1, p2, p3):
    # Check coinciding point/points on the same arc to avoid /0
    # then v1_raw, v2_raw can't be 0 in this case
    # Have checked in sph_area
    v1_raw = fast_cross(fast_cross(p2, p1), p2)
    v2_raw = fast_cross(fast_cross(p2, p3), p2)

    # Special case of antipodal points
    if fast_norm(v1_raw) == 0 or fast_norm(v2_raw) == 0:
        return None

    v1 = v1_raw / fast_norm(v1_raw)
    v2 = v2_raw / fast_norm(v2_raw)

    inprod = fast_dot(v1, v2)
    inprod = fast_clip(inprod)
    return torch.arccos(inprod)


def polygon_area(P):
    N = len(P)
    S = 0
    for i in range(N):
        with torch.no_grad():
            P[i] = P[i] / fast_norm(P[i])
    for i in range(N):
        if sph_angle(P[i], P[(i + 1) % N], P[(i + 2) % N]) == None: return None
        S += sph_angle(P[i], P[(i + 1) % N], P[(i + 2) % N])
    Area = S - (N - 2) * torch.pi
    if Area <= -tol: print('error in polygon_area')
    return S - (N - 2) * torch.pi


def update_polygon(P, N, n):
    l = len(P)
    Tol_upp = 0.0000001
    Tol_bot = 0.0000001
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
    return torch.stack(P_tmp), torch.stack(N_tmp)


def P_intersect(P1, N1, N2):
    '''
    P=[v_1, v_2, ..., v_n] is one polygon (use smaller one, less computation before possible drop)
    N=[n_1, n_2, ..., n_m] is the edge normals of polygon
    P, N: numpy array
    '''
    P_updated = P1.clone()  # .detach().requires_grad_(True)#torch.tensor(P1)
    N1_updated = N1.clone()  # .detach().requires_grad_(True)#torch.tensor(N1)
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
    if (torch.abs(pi - pj) < tol).all():
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
    integral = torch.tensor(0)

    for idx1 in range(len(ECT)):
        for idx2 in range(len(ECT)):
            integral_i = torch.tensor(0)
            integral_j = torch.tensor(0)
            integral_s = torch.tensor(0)
            if ECT[idx1][3] == ECT[idx2][3] and idx1 == idx2:
                P = ECT[idx1][2]
                integral_i = int_polygon(P, ECT[idx1][1])
                integral_s = polygon_area(P)
            elif ECT[idx1][3] == ECT[idx2][3] and idx1 != idx2:
                continue
            else:
                P_divi, P_divj, integral_s = P_div(ECT[idx1][2], ECT[idx2][2], ECT[idx1][4], ECT[idx2][4], ECT[idx1][1],
                                                   ECT[idx2][1])
                if integral_s is not None:
                    if P_divi is not None:
                        integral_i = int_polygon(P_divi, ECT[idx1][1])
                    if P_divj is not None:
                        integral_j = int_polygon(P_divj, ECT[idx2][1])
                else:
                    continue
            integral = integral + ECT[idx1][0] * ECT[idx2][0] * (integral_s - integral_i - integral_j)

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
                    integral_i = int_polygon(P_divi, ECT1[idx1][1])
                if P_divj is not None:
                    integral_j = int_polygon(P_divj, ECT2[idx2][1])
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
            N = return_N(poly)
            megatmp.append(N)
            tmp.append(megatmp)
    return tmp


def return_N(P):
    # Given a polygon, return its edge normals
    l = len(P)
    N = []
    for i in range(l):
        n_raw = fast_cross(P[i], P[(i + 1) % l])
        n = n_raw / fast_norm(n_raw)
        N.append(n)
    N = np.array(N)
    return N
