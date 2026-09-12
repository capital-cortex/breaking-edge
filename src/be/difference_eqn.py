import numpy as np
from typing import Any

class SlidingSystemSolver:
    def __init__(self, n: int, smp_cnt: int) -> None:
        self.n = n
        self.smp_cnt = smp_cnt
        self.k = 2 * n + 1  # Number of features
        self.W = smp_cnt - n  # Effective window size for rows in H
        
        self.P_mat : np.typing.NDArray[np.floating[Any]] # (H^T H)^-1
        self.b_vec : np.typing.NDArray[np.float64]       # H^T * Y
        self.is_initialized = False
        self.last_idx = -1
    
    def _get_row(
            self                                      ,
            x          : np.typing.NDArray[np.float64],
            y          : np.typing.NDArray[np.float64],
            target_idx : int                          ,
    ) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        """Constructs a single row of the H matrix for a given target index."""
        h = np.zeros(self.k)
        # b_coeffs part: -y[i-1] ... -y[i-n]
        for j in range(1, self.n + 1):
            h[j - 1] = -y[target_idx - j]
        # a_coeffs part: x[i] ... x[i-n]
        for j in range(self.n + 1):
            h[self.n + j] = x[target_idx - j]
        return h.reshape(-1, 1)
    
    def update(
            self                               ,
            x   : np.typing.NDArray[np.float64],
            y   : np.typing.NDArray[np.float64],
            idx : int                          ,
    ) -> tuple[np.typing.NDArray[np.float64], np.typing.NDArray[np.float64]]:
        """Slides the window to end at 'idx' and returns (a_coeffs, b_coeffs)."""
        # --- Initialization ---
        if not self.is_initialized:
            # Use your original logic to bootstrap the first window
            xs = x[idx - self.smp_cnt + 1 : idx + 1]
            ys = y[idx - self.smp_cnt + 1 : idx + 1]
            
            # Construct initial H and Y
            rows = self.smp_cnt - self.n
            Y_target = ys[self.n:]
            H = np.zeros((rows, self.k))
            for i in range(rows):
                ni = i + self.n
                H[i, :] = self._get_row(xs, ys, ni).flatten()
            
            # Compute initial P_mat and b_vec
            # Adding a small epsilon (Ridge) helps numerical stability
            self.P_mat = np.linalg.inv(H.T @ H + np.eye(self.k) * 1e-9)
            self.b_vec = H.T @ Y_target
            self.is_initialized = True
            self.last_idx = idx - 1
            
        else:
            assert idx == self.last_idx + 1, "Sliding window must move forward by one index."
            # --- Sliding Logic ---
            # 1. Identify entering and exiting rows
            # The row that exits was the target at (idx - smp_cnt + n)
            out_target_idx = idx - self.smp_cnt + self.n
            h_out          = self._get_row(x, y, out_target_idx)
            y_out : float  = y[out_target_idx]
            
            # The row that enters is the target at (idx)
            h_in = self._get_row(x, y, idx)
            y_in = y[idx]
            
            # 2. Downdate (Remove old data)
            # Sherman-Morrison with negative sign
            denom_out = 1.0 - (h_out.T @ self.P_mat @ h_out).item()
            self.P_mat = self.P_mat + (self.P_mat @ h_out @ h_out.T @ self.P_mat) / denom_out
            self.b_vec = self.b_vec - h_out.flatten() * y_out
            
            # 3. Update (Add new data)
            # Sherman-Morrison with positive sign
            denom_in = 1.0 + (h_in.T @ self.P_mat @ h_in).item()
            self.P_mat = self.P_mat - (self.P_mat @ h_in @ h_in.T @ self.P_mat) / denom_in
            self.b_vec = self.b_vec + h_in.flatten() * y_in
        
        # Solve for theta: (H^T H)^-1 * (H^T Y)
        theta = self.P_mat @ self.b_vec
        self.last_idx = idx
        return theta[self.n:], theta[:self.n]

def solve_system_coeffs(x: np.ndarray, y: np.ndarray, n: int, smp_cnt: int, idx: int):
    # y[i] + b1*y[i-1]... = a0*x[i] + a1*x[i-1]...
    xs = x[idx - smp_cnt + 1 : idx + 1]
    ys = y[idx - smp_cnt + 1 : idx + 1]
    num_samples = len(xs)
    rows = num_samples - n
    Y_target = ys[n:]
    H = np.zeros((rows, 2 * n + 1))
    for i in range(rows):
        ni = i + n
        for j in range(1, n + 1):
            H[i, j - 1] = -ys[ni - j]
        for j in range(n + 1):
            H[i, n + j] = xs[ni - j]
    theta, _, _, _ = np.linalg.lstsq(H, Y_target, rcond=None)
    b_coeffs = theta[:n]
    a_coeffs = theta[n:]
    return a_coeffs, b_coeffs

def get_system_pdn(
        a: np.ndarray,
        b: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        n: int,
        pdn_cnt:int,
        idx: int,
        x_step: int = 0
):
    x_const = np.concatenate([x[idx - n + 1 : idx + 1], np.zeros(pdn_cnt)])
    y_pdn   = np.concatenate([y[idx - n + 1 : idx + 1], np.zeros(pdn_cnt)])
    for i in range(pdn_cnt):
        x_const[n + i] = x_step
        y_pdn  [n + i] = np.dot(a, x_const[i : i + n + 1][::-1]) - np.dot(b, y_pdn[i : i + n][::-1])
    return y_pdn[n:]
