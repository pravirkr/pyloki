import numpy as np
from scipy.optimize import minimize


def fit_initial_polynomial(x, y, degree):
    """Fit initial polynomial and return coefficients and covariance."""
    coeffs = np.polyfit(x, y, degree, cov=True)
    return coeffs[0], coeffs[1]


def polynomial(x, coeffs):
    """Evaluate polynomial at x."""
    return np.polyval(coeffs, x)


def extend_polynomial(x_new, coeffs, cov, num_samples=1000):
    """Generate extended polynomials using covariance matrix."""
    degree = len(coeffs) - 1
    samples = np.random.multivariate_normal(coeffs, cov, num_samples)
    extended_polynomials = [np.polyval(sample, x_new) for sample in samples]
    return np.array(extended_polynomials)


def continuity_differentiability_constraint(new_coeffs, old_coeffs, x_boundary):
    """Enforce continuity and differentiability at the boundary."""
    old_val = polynomial(x_boundary, old_coeffs)
    new_val = polynomial(x_boundary, new_coeffs)
    old_deriv = np.polyder(old_coeffs)(x_boundary)
    new_deriv = np.polyder(new_coeffs)(x_boundary)
    return (old_val - new_val) ** 2 + (old_deriv - new_deriv) ** 2


def extend_with_constraints(x_old, y_old, x_new, degree):
    """Extend polynomial fit with continuity and differentiability constraints."""
    old_coeffs, cov = fit_initial_polynomial(x_old, y_old, degree)
    x_boundary = x_old[-1]

    def objective(new_coeffs):
        return continuity_differentiability_constraint(
            new_coeffs, old_coeffs, x_boundary
        )

    extended_coeffs = []
    for _ in range(100):  # Generate 100 valid extensions
        initial_guess = np.random.multivariate_normal(old_coeffs, cov)
        result = minimize(objective, initial_guess, method="BFGS")
        if result.success:
            extended_coeffs.append(result.x)

    return np.array(extended_coeffs)


# Example usage
np.random.seed(42)
x_old = np.linspace(0, 5, 20)
y_old = 2 * x_old**2 - 3 * x_old + 1 + np.random.normal(0, 0.5, 20)
x_new = np.linspace(5, 10, 20)
degree = 2

extended_coeffs = extend_with_constraints(x_old, y_old, x_new, degree)

# Plot results
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.scatter(x_old, y_old, color="blue", label="Original Data")
plt.plot(
    x_old,
    polynomial(x_old, np.polyfit(x_old, y_old, degree)),
    "r-",
    label="Initial Fit",
)

for coeffs in extended_coeffs[:10]:  # Plot first 10 extended polynomials
    plt.plot(x_new, polynomial(x_new, coeffs), "g-", alpha=0.1)

plt.plot([], [], "g-", label="Extended Polynomials")
plt.legend()
plt.title("Polynomial Extension with Continuity and Differentiability")
plt.xlabel("x")
plt.ylabel("y")
plt.show()
