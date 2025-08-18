# flux.py
import numpy as np
from common.definitions import (idx_2d_rho, idx_2d_rho_u, idx_2d_rho_w, idx_2d_rho_theta)

def roe_flux_2d_delta0(
    U_L, U_R,                        # (neq, N)
    u_L, u_R,                        # (N,)
    w_L, w_R,                        # (N,)
    p_L, p_R,                        # (N,)
    H_L, H_R,                        # (N,)
    h_L, h_R,                        # (N,)
    *,
    gamma: float,
    gravity: float,
    normal: str = "z"                # "z" for kfaces; "x" for ifaces
    ):
    """Standard Roe interfacial flux (δ=0) for 2D Euler + gravity."""
    assert normal in ("z", "x")
    neq, N = U_L.shape
    psi = gamma - 1.0

    # Coerce any tiny complex noise to real
    def real(x): return np.real_if_close(x, tol=1000)

    U_L = real(U_L); U_R = real(U_R)
    u_L = real(u_L); u_R = real(u_R)
    w_L = real(w_L); w_R = real(w_R)
    p_L = real(p_L); p_R = real(p_R)
    H_L = real(H_L); H_R = real(H_R)
    h_L = real(h_L); h_R = real(h_R)

    rho_L = U_L[idx_2d_rho]; rho_R = U_R[idx_2d_rho]

    # Physical fluxes on each side
    if normal == "z":
        vL_n, vR_n = w_L, w_R
        mom_n_idx = idx_2d_rho_w
    else:
        vL_n, vR_n = u_L, u_R
        mom_n_idx = idx_2d_rho_u

    F_L = U_L * vL_n
    F_L[idx_2d_rho_theta] += p_L * vL_n
    F_L[mom_n_idx]        += p_L

    F_R = U_R * vR_n
    F_R[idx_2d_rho_theta] += p_R * vR_n
    F_R[mom_n_idx]        += p_R

    # Roe averages
    RT       = np.sqrt(rho_R / rho_L)
    rho_avg  = RT * rho_L
    u_avg    = (RT * u_R + u_L) / (RT + 1.0)
    w_avg    = (RT * w_R + w_L) / (RT + 1.0)
    H_avg    = (RT * H_R + H_L) / (RT + 1.0)
    h_avg    = 0.5 * (h_R + h_L)

    c2 = psi * (H_avg - 0.5 * (u_avg**2 + w_avg**2) - gravity * h_avg)
    c2 = np.maximum(c2, 0.0)   # guard roundoff
    c  = np.sqrt(c2)

    vn     = w_avg if normal == "z" else u_avg
    lam_p  = vn + c
    lam_m  = vn - c
    abs_p  = np.abs(lam_p)
    abs_m  = np.abs(lam_m)
    abs_vn = np.abs(vn)

    eps = 1e-14
    S1 = -(rho_avg / (2.0 * (c + eps))) * (abs_m - abs_p)
    S2 = -(1.0 / (2.0 * rho_avg * (c + eps))) * (abs_m - abs_p)
    S3 = 0.5 * (abs_p + abs_m)

    alph1 = 0.5 * (u_avg**2 + w_avg**2)

    zeta1 = (S3 - abs_vn) / (c2 + eps)
    zeta2 = S2 * rho_avg + vn * zeta1
    zeta3 = (S3 * rho_avg + S1 * vn) / rho_avg
    zeta4 = (S3 / psi) + alph1 * zeta1 + S2 * rho_avg * vn
    zeta5 = (S3 * rho_avg * vn + (S1 * c2) / psi + S1 * alph1) / rho_avg

    D = np.zeros((N, neq, neq), dtype=U_L.dtype)

    if normal == "z":
        D[:, 0, 0] = abs_vn - (S1 * vn) / rho_avg + alph1 * psi * zeta1
        D[:, 0, idx_2d_rho_u] = -psi * u_avg * zeta1
        D[:, 0, idx_2d_rho_w] = -psi * vn * zeta1 + (S1 / rho_avg)
        D[:, 0, idx_2d_rho_theta] = psi * zeta1

        D[:, 1, 0] = alph1 * psi * u_avg * zeta1 - (S1 * u_avg * vn) / rho_avg
        D[:, 1, idx_2d_rho_u] = abs_vn - psi * u_avg**2 * zeta1
        D[:, 1, idx_2d_rho_w] = (S1 * u_avg) / rho_avg - psi * u_avg * vn * zeta1
        D[:, 1, idx_2d_rho_theta] = psi * u_avg * zeta1

        D[:, 2, 0] = vn * abs_vn + alph1 * psi * zeta2 - vn * zeta3
        D[:, 2, idx_2d_rho_u] = -psi * u_avg * zeta2
        D[:, 2, idx_2d_rho_w] = -vn * psi * zeta2 + zeta3
        D[:, 2, idx_2d_rho_theta] = psi * zeta2

        D[:, 3, 0] = alph1 * abs_vn - vn * zeta5 - u_avg**2 * abs_vn + alph1 * psi * zeta4
        D[:, 3, idx_2d_rho_u] = u_avg * abs_vn - psi * u_avg * zeta4
        D[:, 3, idx_2d_rho_w] = zeta5 - psi * vn * zeta4
        D[:, 3, idx_2d_rho_theta] = psi * zeta4
    else:
        D[:, 0, 0] = abs_vn - (S1 * vn) / rho_avg + alph1 * psi * zeta1
        D[:, 0, idx_2d_rho_u] = -psi * vn * zeta1 + (S1 / rho_avg)
        D[:, 0, idx_2d_rho_w] = -psi * w_avg * zeta1
        D[:, 0, idx_2d_rho_theta] = psi * zeta1

        D[:, 1, 0] = vn * abs_vn + alph1 * psi * zeta2 - vn * zeta3
        D[:, 1, idx_2d_rho_u] = -vn * psi * zeta2 + zeta3
        D[:, 1, idx_2d_rho_w] = -psi * w_avg * zeta2
        D[:, 1, idx_2d_rho_theta] = psi * zeta2

        D[:, 2, 0] = alph1 * psi * w_avg * zeta1 - (S1 * u_avg * w_avg) / rho_avg
        D[:, 2, idx_2d_rho_u] = (S1 * w_avg) / rho_avg - psi * vn * w_avg * zeta1
        D[:, 2, idx_2d_rho_w] = abs_vn - psi * w_avg**2 * zeta1
        D[:, 2, idx_2d_rho_theta] = psi * w_avg * zeta1

        D[:, 3, 0] = alph1 * abs_vn - vn * zeta5 - w_avg**2 * abs_vn + alph1 * psi * zeta4
        D[:, 3, idx_2d_rho_u] = zeta5 - psi * vn * zeta4
        D[:, 3, idx_2d_rho_w] = w_avg * abs_vn - psi * w_avg * zeta4
        D[:, 3, idx_2d_rho_theta] = psi * zeta4

    dU  = U_R - U_L
    phi = np.array([ D[i].dot(dU[:, i]) for i in range(N) ])  # (N, neq)

    return 0.5 * (F_L + F_R) - 0.5 * phi.T
