import numpy as np
from scipy.special import comb


def compute_connection_coefficient_r(k: int, m: int) -> float:
    if k < 0 or m < 0 or m > k or (k - m) % 2 != 0:
        return 0.0
    if m == 0 and k % 2 == 0:
        return (-1) ** (k // 2)
    n = (k - m) // 2
    r = (k + m) // 2
    return ((-1) ** n) * (2 ** (m - 1)) * (2 * k / (k + m)) * comb(r, n)


def compute_connection_coefficient_s(k: int, m: int) -> float:
    # Check if k-m is even and m <= k
    if k < 0 or m < 0 or m > k or (k - m) % 2 != 0:
        return 0.0
    if k == 0 and m == 0:
        return 1.0
    n = (k - m) // 2
    deltam0 = 1 if m == 0 else 0
    return 2 ** (1 - k - deltam0) * comb(k, n)


def compute_transformation_coefficient_c(n: int, k: int, p: float, q: float) -> float:
    if k > n:
        return 0.0
    result = 0.0
    for m in range(k, n + 1):
        r_nm = compute_connection_coefficient_r(n, m)
        inner_sum = 0.0
        for i in range(k, m + 1):
            s_ik = compute_connection_coefficient_s(i, k)
            term = comb(m, i) * p**i * q ** (m - i) * s_ik
            inner_sum += term
        result += r_nm * inner_sum
    return result


def poly_chebyshev_transform_matrix(
    poly_order: int,
    tc1: float,
    ts1: float,
    tc2: float,
    ts2: float,
) -> np.ndarray:
    k_max = poly_order
    if k_max < 0:
        msg = "k_max must be a non-negative integer."
        raise ValueError(msg)
    if ts1 <= 0 or ts2 <= 0:
        msg = "ts1 and ts2 must be positive."
        raise ValueError(msg)
    p = ts2 / ts1
    q = (tc2 - tc1) / ts1
    c_mat = np.zeros((k_max + 1, k_max + 1), dtype=np.float32)
    for n in range(k_max + 1):
        for k in range(k_max + 1):
            c_mat[n, k] = compute_transformation_coefficient_c(n, k, p, q)
    return c_mat


def power_pols(n, t_0):
    pols = np.zeros([n + 1, n + 1])
    for i in range(n + 1):
        for j in range(i + 1):
            pols[i, j] = comb(i, j) * (-t_0) ** (i - j)
    return pols


def translate_pols(pols, t_0):
    poly_order = len(pols[0]) - 1
    shifting_pols = power_pols(poly_order, t_0)
    return np.dot(pols, shifting_pols)


def scale_pols(pols, scale):
    output = np.zeros(pols.shape)
    for j in range(len(pols)):
        for i in range(len(pols[j])):
            output[j, i] = pols[j, i] / scale**i
    return output


def generate_chebyshev_polys_table(order, n_derivatives):
    tab = np.zeros((n_derivatives + 1, order + 1, order + 1))
    tab[0][0][0] = 1.0
    tab[0][1][1] = 1.0
    for i in range(2, order + 1):
        tab[0][i] = 2 * np.roll(tab[0][i - 1], 1) - tab[0][i - 2]
    for i in range(1, order + 1):
        for j in range(1, n_derivatives + 1):
            tab[j][i] = np.roll(tab[j - 1][i], -1) * np.arange(1, order + 2)
            tab[j][i][-1] = 0.0
    return tab


def generalized_cheb_pols(poly_order, t0, scale):
    cheb_pols = generate_chebyshev_polys_table(poly_order, 0)[0]
    pols_scaled = scale_pols(cheb_pols, scale)
    pols_trans = translate_pols(pols_scaled, t0)
    return pols_trans


def gen_transfer_matrix(poly_order, tc1, ts1, tc2, ts2):
    M1 = generalized_cheb_pols(poly_order, tc1, ts1)
    M2 = generalized_cheb_pols(poly_order, tc2, ts2)
    return np.dot(M1, np.linalg.inv(M2))


poly_order = 4
tc1, ts1 = 5, 1
tc2, ts2 = 10, 2

mat_inv = gen_transfer_matrix(poly_order, tc1, ts1, tc2, ts2)
mat_direct = poly_chebyshev_transform_matrix(poly_order, tc1, ts1, tc2, ts2)
