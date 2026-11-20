from typing import Tuple
import numpy as np
import csv
from scipy.linalg import cho_factor, cho_solve

def vectorize(M:np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    Stacks columns one above the other.

    :param M: Matrix to vectorize
    :return: Vectorized matrix, n (shape[0]), m (shape[1])
    """
    return M.flatten(order="F").reshape(-1, 1), M.shape[0], M.shape[1]


def reshape(V:np.ndarray, n:float, m:float) -> np.ndarray:
    """
    Reshape a vector to a matrix of shape (n, m)

    Inverse of vectorize(): M = reshape(*vectorize(M))

    :param V: Vector to reshape
    :param n: shape[0]
    :param m: shape[1]
    :return: Reshaped matrix
    """
    if len(V.shape) == 2:
        if V.shape[1] > 1:
            txt = f"Input vector must be 1-dimensional, got {V.shape}"
            raise ValueError(txt)
    if V.shape[0] != n*m:
        txt = f"Input vector lenght must be {n*m}, got {V.shape[0]}"
        raise ValueError(txt)
    return V.reshape((n, m), order="F")


def ols(G:np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Y = G @ X
    X = (G.T @ G)-1 @ G.T @ Y
    Returns: X
    """
    return np.linalg.inv(G.transpose() @ G) @ G.transpose() @ Y


def wtls(Y: np.ndarray, Q_y: np.ndarray, G: np.ndarray, Q_g: np.ndarray,
         threshold: float = 1e-4, max_iterations: int = 20
         ) -> tuple[np.ndarray, np.ndarray]:
    """Q_g : (n, m) variances de chaque élément de G, ou (n*m, n*m) covariance pleine de vec(G)."""
    y = np.asarray(Y, float).ravel()
    G = np.asarray(G, float)
    Q_g = np.asarray(Q_g, float)
    n, m = G.shape
    diag = Q_g.shape == (n, m)
    if not diag:
        Qg4 = Q_g.reshape(m, n, m, n)          # Qg4[j, i, k, l] = Cov(G[i, j], G[l, k])

    def corrections(x):
        r = y - G @ x
        if diag:
            Q = Q_y + np.diag(Q_g @ x**2)
        else:
            T = np.tensordot(Qg4, x, axes=([2], [0]))   # blocs de Q_g X
            Q = Q_y + np.tensordot(x, T, axes=1)         # Xᵀ Q_g X
        cQ = cho_factor(Q)
        lam = cho_solve(cQ, r)
        E_g = -Q_g * np.outer(lam, x) if diag else -(T @ lam).T
        return r, cQ, lam, E_g

    WG = cho_solve(cho_factor(Q_y), G)
    x = np.linalg.solve(G.T @ WG, WG.T @ y)

    for _ in range(max_iterations):
        _, cQ, _, E_g = corrections(x)
        G_hat = G - E_g
        WG = cho_solve(cQ, G_hat)
        x_new = np.linalg.solve(G_hat.T @ WG, WG.T @ (y - E_g @ x))
        converged = np.linalg.norm(x_new - x) < threshold
        x = x_new
        if converged:
            break

    r, cQ, lam, E_g = corrections(x)
    G_hat = G - E_g
    N = G_hat.T @ cho_solve(cQ, G_hat)
    sigma_2 = (r @ lam) / (n - m)
    Q_x = sigma_2 * cho_solve(cho_factor(N), np.eye(m))

    return (x[:, None] if np.ndim(Y) == 2 else x), Q_x


def old_wtls(Y:np.ndarray, Q_y:np.ndarray, G:np.ndarray, Q_g:np.ndarray, threshold=1e-4, max_iterations=20) \
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Optimized with Claude AI.
    
    Weighted Total Least Squares (WTLS).

    Estimates the parameter vector x of the model Y ≈ G·x when both the observations Y **and** the design matrix G are
    subject to uncertainty (errors-in-variables model). Unlike ordinary weighted least squares (WLS), which assumes G is
    exact, this method corrects Y and G simultaneously.

    The solution is iterative (Gauss-Newton-type linearization). At each iteration, the effective residual covariance
    Q = Q_y + Xᵀ Q_g X is computed, then the Lagrange multiplier λ = Q⁻¹ r, then the correction to the design matrix
    (E_g), and finally a WLS problem is solved on the corrected matrix G_hat = G - E_g.

    Notes
    -----
    - The initial guess is the classical WLS solution (G assumed exact).
    - Requires n > m (more observations than parameters); otherwise σ² is undefined (division by n - m).
    - Convergence is not guaranteed if the uncertainties on G are very large relative to the values of G.
    - No warning is raised if the algorithm has not converged after `max_iterations`: the last estimate is returned as
    is.
        
    :param Y: Observation vector, shape (n,) or (n, 1).
    :param Q_y: Covariance matrix of the observations Y (symmetric positive definite), shape (n, n)
    :param G: Design matrix (n observations, m parameters), known with uncertainty, shape (n, m)
    :param Q_g: Uncertainty of G, given in one of two forms:
                - shape (n, m): variance of each element of G (elements assumed uncorrelated);
                - shape (n*m, n*m): full covariance of vec(G), which allows correlations between elements to be modeled.
    :param threshold: Convergence criterion: stop when ‖x_new - x‖ < threshold.
    :param max_iterations: Maximum number of iterations.
    :return: x : Estimated parameters. Shape (m, 1) if Y is 2-D, (m,) otherwise.
             Q_x : A posteriori covariance matrix of the estimated parameters, shape (m, m)
    """
    y = np.asarray(Y, float).ravel()
    G = np.asarray(G, float)
    Q_g = np.asarray(Q_g, float)
    n, m = G.shape
    diag = Q_g.shape == (n, m)
    if not diag:
        Qg4 = Q_g.reshape(m, n, m, n)

    def corrections(x):
        r = y - G @ x
        if diag:
            Q = Q_y + np.diag(Q_g @ x**2)
        else:
            T = np.tensordot(Qg4, x, axes=([2], [0]))
            Q = Q_y + np.tensordot(x, T, axes=1)
        lam = np.linalg.solve(Q, r)
        E_g = -Q_g * np.outer(lam, x) if diag else -(T @ lam).T
        return r, Q, lam, E_g

    WG = np.linalg.solve(Q_y, G)
    x = np.linalg.solve(G.T @ WG, WG.T @ y)

    for _ in range(max_iterations):
        _, Q, _, E_g = corrections(x)
        G_hat = G - E_g
        WG = np.linalg.solve(Q, G_hat)
        x_new = np.linalg.solve(G_hat.T @ WG, WG.T @ (y - E_g @ x))
        converged = np.linalg.norm(x_new - x) < threshold
        x = x_new
        if converged:
            break

    r, Q, lam, E_g = corrections(x)
    G_hat = G - E_g
    N = G_hat.T @ np.linalg.solve(Q, G_hat)
    sigma_2 = (r @ lam) / (n - m)
    Q_x = sigma_2 * np.linalg.inv(N)

    return (x[:, None] if np.ndim(Y) == 2 else x), Q_x


def wtls_2D(Y:np.ndarray, Q_y:np.ndarray, G:np.ndarray, Q_g:np.ndarray, threshold=1e-4, max_iterations=20) \
        -> Tuple[np.ndarray, np.ndarray]:
    X_list = []
    Q_x_list = []
    for i in range(Y.shape[1]):
        if Q_y.ndim == 2: # Q_y contains the variances of Y
            local_Q_y = np.diag(Q_y[:, i])
        elif Q_y.ndim == 3: # Q_y contains the variance-covariances of Y
            local_Q_y = Q_y[i]
        else:
            raise TypeError("Q_y must either be a 2d variance or a 3d variance-covariance matrix.")
        local_Y = Y[:, i].reshape(-1, 1)
        local_X, local_Q_x = wtls(Y=local_Y, Q_y=local_Q_y, G=G, Q_g=Q_g,
                                  threshold=threshold, max_iterations=max_iterations)

        X_list.append(local_X)
        Q_x_list.append(local_Q_x)

    X = np.hstack(X_list).transpose()
    Q_x = np.stack(Q_x_list)

    return X, Q_x



def cov_mediane(Q: list|np.ndarray) -> np.ndarray:
    """Q : liste de N matrices de covariance (p, p), une par vecteur ou matrice (N, p, p)"""
    if type(Q) is list:
        Q = np.stack([np.asarray(S, dtype=float) for S in Q])  # (N, p, p)
    else:
        if type(Q) is not np.ndarray:
            raise TypeError("Q must be a list or numpy array")
        if Q.shape[1] != Q.shape[2]:
            raise ValueError("Q must be of shape (N, p, p)")

    sig = np.sqrt(np.diagonal(Q, axis1=1, axis2=2))             # (N, p)
    rho = Q / (sig[:, :, None] * sig[:, None, :])                # (N, p, p)
    num = np.arcsin(np.clip(rho, -1, 1)).sum(axis=0)                  # (p, p)
    S = (1 / sig).sum(axis=0)                                         # (p,)
    return num / np.outer(S, S)

def cov_arctan2(a, b, var_a, var_b, cov_ab=0.0):
    r2 = a**2 + b**2
    return (b**2 * var_a + a**2 * var_b - 2 * a * b * cov_ab) / r2**2


def save_dict_to_csv(d: dict, filepath: str):
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(d.keys())
        writer.writerows(zip(*d.values()))


def load_dict_from_csv(filepath: str) -> dict:
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        d = {k: [] for k in reader.fieldnames}
        for row in reader:
            for k, v in row.items():
                d[k].append(v)
    return d