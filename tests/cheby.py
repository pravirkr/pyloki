import sympy as sp


def r_nm(n: int, m: int) -> int:
    """Compute R_{n,m} coefficient for Chebyshev expansion."""
    if n < 0 or m < 0 or m > n or (n - m) % 2 != 0:
        return 0
    if m == 0 and n % 2 == 0:
        return (-1) ** (n // 2)
    k = (n - m) // 2
    r = (n + m) // 2
    return ((-1) ** k) * (2 ** (m - 1)) * sp.Rational(2 * n, n + m) * sp.binomial(r, k)


def s_ik(i: int, k: int) -> int:
    """Compute S_{i,k} coefficient for power to Chebyshev conversion."""
    if i < 0 or k < 0 or k > i or (i - k) % 2 != 0:
        return 0
    delta_k0 = 1 if k == 0 else 0
    return 2 ** (1 - i - delta_k0) * sp.binomial(i, (i - k) // 2)


def c_nk(n: int, k: int, p: int, q: int) -> int:
    """Compute C_{n,k}(p,q) coefficient."""
    result = 0
    for m in range(k, n + 1):
        for i in range(k, m + 1):
            term = r_nm(n, m) * sp.binomial(m, i) * p**i * q ** (m - i) * s_ik(i, k)
            result += term
    return sp.simplify(result)


# Direct verification by computing T_n(px + q)
def cheb_t(n: int, x: sp.Symbol) -> sp.Expr:
    if n == 0:
        return 1.0
    if n == 1:
        return x
    return 2.0 * x * cheb_t(n - 1, x) - cheb_t(n - 2, x)


def verify_coefficients_r(max_n: int) -> None:
    x = sp.Symbol("x")

    for n in range(max_n + 1):
        # Generate using both methods
        poly1 = cheb_t(n, x)
        poly2 = sum(r_nm(n, m) * x**m for m in range(n + 1))
        diff = sp.simplify(poly1 - poly2)
        if diff == 0:
            print(f"T_{n}(x): Match!")
        else:
            print(f"T_{n}(x): Mismatch!")
            print(f"Recurrence: {poly1}")
            print(f"R-coeffs:   {poly2}")
            print(f"Difference: {diff}")
            print()


# Run verification
verify_coefficients_r(17)  # Verify up to T_5(x)

# Define symbols
p, q = sp.symbols("p q")

# Print coefficients for n=2,3,4
for n in range(2, 5):
    print(f"\nFor n = {n}:")
    for k in range(n + 1):
        coeff = c_nk(n, k, p, q)
        print(f"C_{{{n},{k}}}(p,q) = {coeff}")

# Add this to the verification section
x = sp.Symbol("x")
for n in range(2, 5):
    direct = cheb_t(n, p * x + q)
    via_coeff = sum(c_nk(n, k, p, q) * cheb_t(k, x) for k in range(n + 1))
    print(f"\nT_{n} verification:")
    print(f"Direct: {sp.expand(direct)}")
    print(f"Via coefficients: {sp.expand(via_coeff)}")
    print(f"Difference: {sp.simplify(direct - via_coeff)}")
    print(f"Match: {sp.simplify(direct - via_coeff) == 0}")
