#region imports
import Gauss_Elim as GE
from math import sqrt, pi, exp, cos
#endregion


#region function definitions
def Probability(PDF, args, c, GT=True):
    """
    Compute probability for a Normal(μ, σ) using Simpson's rule:
      - If GT=False: return P(x < c)
      - If GT=True : return P(x > c) = 1 - P(x < c)

    PDF is a callback that takes a SINGLE tuple argument: (x, mu, sig). :contentReference[oaicite:8]{index=8}
    args is (mu, sig). :contentReference[oaicite:9]{index=9}
    Integrate from (mu - 5*sig) to c for P(x<c). :contentReference[oaicite:10]{index=10}
    """
    mu, sig = args
    lhl = mu - 5.0 * sig
    rhl = c

    # quick clamps (helps if c is far outside)
    if rhl <= lhl:
        p_lt = 0.0
    elif rhl >= mu + 5.0 * sig:
        p_lt = 1.0
    else:
        p_lt = Simpson(PDF, (mu, sig, lhl, rhl), N=200)  # N even; higher N = better accuracy

    return (1.0 - p_lt) if GT else p_lt


def GPDF(args):
    """
    Gaussian/Normal PDF
    args = (x, mu, sig)
    """
    x, mu, sig = args
    return (1.0 / (sig * sqrt(2.0 * pi))) * exp(-0.5 * ((x - mu) / sig) ** 2)


def Simpson(fn, args, N=100):
    """
    Simpson's 1/3 rule integration.
    args = (mu, sig, lhl, rhl)
    """
    mu, sig, lhl, rhl = args

    if N < 2:
        N = 2
    if N % 2 == 1:
        N += 1  # Simpson requires even N

    h = (rhl - lhl) / N

    total = 0.0
    for i in range(N + 1):
        x = lhl + i * h
        fx = fn((x, mu, sig))

        if i == 0 or i == N:
            total += fx
        elif i % 2 == 1:
            total += 4.0 * fx
        else:
            total += 2.0 * fx

    return (h / 3.0) * total


def Secant(fcn, x0, x1, maxiter=10, xtol=1e-5):
    """
    Secant Method:
    x_new = x1 - f(x1)*(x1-x0)/(f(x1)-f(x0))
    Stop if |x_new - x1| < xtol OR reached maxiter.
    Return (root_estimate, iterations_used) :contentReference[oaicite:11]{index=11}
    """
    # We will count how many NEW x values we compute
    for it in range(1, maxiter + 1):
        f0 = fcn(x0)
        f1 = fcn(x1)

        denom = (f1 - f0)
        if abs(denom) < 1e-14:
            # can't divide safely; return current estimate
            return x1, it - 1

        x2 = x1 - f1 * (x1 - x0) / denom

        if abs(x2 - x1) < xtol:
            return x2, it

        x0, x1 = x1, x2

    return x1, maxiter


def GaussSeidel(Aaug, x, Niter=15):
    """
    Gauss-Seidel iterative solver for augmented matrix [A|b]. :contentReference[oaicite:12]{index=12}
    Aaug is N rows x (N+1) cols.
    x is initial guess vector length N.
    """
    # Step 1: make diagonal dominant (row reordering) :contentReference[oaicite:13]{index=13}
    Aaug = GE.MakeDiagDom(Aaug)

    n = len(Aaug)

    for k in range(Niter):
        for i in range(n):
            # b is last column
            rhs = Aaug[i][n]

            # subtract sum(A[i][j]*x[j]) for j != i
            for j in range(n):
                if j != i:
                    rhs -= Aaug[i][j] * x[j]

            # solve for x[i]
            x[i] = rhs / Aaug[i][i]

    return x

#endregion
