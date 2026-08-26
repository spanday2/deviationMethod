import numpy
import sys
import pdb

from common.definitions import idx_rho_u1, idx_rho_u2, idx_rho_w, idx_rho, idx_rho_theta, gravity, p0, Rd, cpd, cvd, heat_capacity_ratio

# For type hints
from common.parallel import DistributedWorld
from geometry        import CubedSphere, DFROperators, Metric3DTopo
from init.dcmip      import dcmip_schar_damping, dcmip_gravity_wave, dcmip_steady_state_mountain


# ============================================================
# Riemann-solver selection
# ============================================================
# Valid values:
#   "rusanov"   : local Lax-Friedrichs/Rusanov flux
#   "ausmplusup": AUSM+up flux
RIEMANN_SOLVER = "rusanov"  # Default Riemann solver for Euler equations


def _validate_riemann_solver(name):
    name = name.lower()
    valid = {"rusanov", "ausmplusup"}
    if name not in valid:
        raise ValueError(
            f"Unknown RIEMANN_SOLVER={name!r}. Expected one of {sorted(valid)}."
        )
    return name

def ausm_3d_vert(
    variables_itf_k, pressure_itf_k, metric,
    flux_x3_itf_k, wflux_adv_x3_itf_k, wflux_pres_x3_itf_k,
    nb_interfaces_vert
):

    beta = 0.125      # 1/8
    K_p = 0.25
    K_u = 0.75
    sigma = 1.0

    # Low-Mach cutoffs
    M_INF_P = 0.1
    M_INF_U = 1e-12

    for itf in range(nb_interfaces_vert):

        elem_D = itf
        elem_U = itf + 1

        sqrtG_face = metric.sqrtG_itf_k[itf, :, :]
        h31_face = metric.H_contra_31_itf_k[itf, :, :]
        h32_face = metric.H_contra_32_itf_k[itf, :, :]
        h33_face = metric.H_contra_33_itf_k[itf, :, :]

        rho_D = variables_itf_k[idx_rho, :, elem_D, 1, :]
        rho_U = variables_itf_k[idx_rho, :, elem_U, 0, :]

        w_D = variables_itf_k[idx_rho_w, :, elem_D, 1, :] / numpy.maximum(rho_D, 1.0e-12)
        w_U = variables_itf_k[idx_rho_w, :, elem_U, 0, :] / numpy.maximum(rho_U, 1.0e-12)

        p_D = pressure_itf_k[:, elem_D, 1, :]
        p_U = pressure_itf_k[:, elem_U, 0, :]

        a_D = numpy.sqrt(h33_face * heat_capacity_ratio * p_D / numpy.maximum(rho_D, 1.0e-12))
        a_U = numpy.sqrt(h33_face * heat_capacity_ratio * p_U / numpy.maximum(rho_U, 1.0e-12))

        a_half = 0.5 * (a_D + a_U)

        M_D = w_D / numpy.maximum(a_D, 1.0e-14)
        M_U = w_U / numpy.maximum(a_U, 1.0e-14)

        M_D = numpy.nan_to_num(M_D, nan=0.0, posinf=0.0, neginf=0.0)
        M_U = numpy.nan_to_num(M_U, nan=0.0, posinf=0.0, neginf=0.0)

        M_bar_sq = 0.5 * (M_D**2 + M_U**2)
        M_bar = numpy.sqrt(numpy.maximum(M_bar_sq, 0.0))

        M_0_p = numpy.minimum(1.0, numpy.maximum(M_bar, M_INF_P))
        fa_p = M_0_p * (2.0 - M_0_p)

        M_0_u = numpy.minimum(1.0, numpy.maximum(M_bar, M_INF_U))
        fa_u = M_0_u * (2.0 - M_0_u)

        alpha = 0.1875 * (-4.0 + 5.0 * fa_p**2)

        M_D_plus = 0.25 * (M_D + 1.0)**2 * (1.0 + 4.0 * beta * (M_D - 1.0)**2)
        M_U_minus = -0.25 * (M_U - 1.0)**2 * (1.0 + 4.0 * beta * (M_U + 1.0)**2)

        # ============================================================
        # Physical sound speeds. Used ONLY in Mp normalization.
        # ============================================================

        c_D = numpy.sqrt(heat_capacity_ratio * p_D / numpy.maximum(rho_D, 1.0e-12))
        c_U = numpy.sqrt(heat_capacity_ratio * p_U / numpy.maximum(rho_U, 1.0e-12))
        c_half = 0.5 * (c_D + c_U)

        rho_half = 0.5 * (rho_D + rho_U)

        Mp = -(K_p / numpy.maximum(fa_p, 1.0e-12)) * numpy.maximum(1.0 - sigma * M_bar_sq, 0.0) * (p_U - p_D) / numpy.maximum(rho_half * c_half**2, 1.0e-12)
        M = M_D_plus + M_U_minus + Mp

        P_D_plus_coeff = 0.25 * (M_D + 1.0)**2 * (2.0 - M_D + 4.0 * alpha * M_D * (M_D - 1.0)**2)
        P_U_minus_coeff = 0.25 * (M_U - 1.0)**2 * (2.0 + M_U - 4.0 * alpha * M_U * (M_U + 1.0)**2)

        P_D_plus = P_D_plus_coeff * p_D
        P_U_minus = P_U_minus_coeff * p_U

        Pw = -K_u * P_D_plus_coeff * P_U_minus_coeff * (rho_D + rho_U) * fa_u * a_half * (w_U - w_D)
        P = P_D_plus + P_U_minus + Pw

        # ============================================================
        # vertical numerical flux
        # ============================================================

        adv_flux = sqrtG_face * (numpy.maximum(0.0, M) * a_D * variables_itf_k[:, :, elem_D, 1, :] + numpy.minimum(0.0, M) * a_U * variables_itf_k[:, :, elem_U, 0, :])
        flux_x3_itf_k[:, :, elem_D, 1, :] = adv_flux

        flux_x3_itf_k[idx_rho_u1, :, elem_D, 1, :] += h31_face * sqrtG_face * P
        flux_x3_itf_k[idx_rho_u2, :, elem_D, 1, :] += h32_face * sqrtG_face * P
        flux_x3_itf_k[idx_rho_w, :, elem_D, 1, :] += h33_face * sqrtG_face * P

        # Same interface flux on both sides
        flux_x3_itf_k[:, :, elem_U, 0, :] = flux_x3_itf_k[:, :, elem_D, 1, :]

        # ============================================================
        # Separate rho-w advective contribution
        # ============================================================

        wflux_adv_face = adv_flux[idx_rho_w, :, :]
        wflux_adv_x3_itf_k[:, elem_D, 1, :] = wflux_adv_face
        wflux_adv_x3_itf_k[:, elem_U, 0, :] = wflux_adv_face

        # ============================================================
        # Separate rho-w pressure contribution
        # ============================================================

        wflux_pres_face = h33_face * sqrtG_face * P
        wflux_pres_x3_itf_k[:, elem_D, 1, :] = wflux_pres_face / numpy.maximum(p_D, 1.0e-12)
        wflux_pres_x3_itf_k[:, elem_U, 0, :] = wflux_pres_face / numpy.maximum(p_U, 1.0e-12)


def ausm_3d_hori_ausmplusup(
    variables_itf_i, pressure_itf_i, u1_itf_i,
    variables_itf_j, pressure_itf_j, u2_itf_j,
    metric,
    flux_x1_itf_i, wflux_adv_x1_itf_i, wflux_pres_x1_itf_i,
    flux_x2_itf_j, wflux_adv_x2_itf_j, wflux_pres_x2_itf_j,
    nb_interfaces_hori, idx_rho, idx_rho_u1, idx_rho_u2, idx_rho_w,
    heat_capacity_ratio
):

    beta = 0.125
    K_p = 0.25
    K_u = 0.75
    sigma = 1.0

    # Low-Mach cutoffs
    M_INF_P = 0.1
    M_INF_U = 1e-12

    for itf in range(nb_interfaces_hori):

        elem_L = itf
        elem_R = itf + 1

        # ============================================================
        # X1 DIRECTION
        # ============================================================

        sqrtG_i = metric.sqrtG_itf_i[:, :, itf]
        h11 = metric.H_contra_11_itf_i[:, :, itf]
        h12 = metric.H_contra_12_itf_i[:, :, itf]
        h13 = metric.H_contra_13_itf_i[:, :, itf]

        rho_L = variables_itf_i[idx_rho, :, elem_L, 1, :]
        rho_R = variables_itf_i[idx_rho, :, elem_R, 0, :]

        p_L = pressure_itf_i[:, elem_L, 1, :]
        p_R = pressure_itf_i[:, elem_R, 0, :]

        u_L = u1_itf_i[:, elem_L, 1, :]
        u_R = u1_itf_i[:, elem_R, 0, :]

        a_L = numpy.sqrt(h11 * heat_capacity_ratio * p_L / numpy.maximum(rho_L, 1.0e-12))
        a_R = numpy.sqrt(h11 * heat_capacity_ratio * p_R / numpy.maximum(rho_R, 1.0e-12))
        a_half = 0.5 * (a_L + a_R)

        M_L = u_L / numpy.maximum(a_L, 1.0e-14)
        M_R = u_R / numpy.maximum(a_R, 1.0e-14)

        M_L = numpy.nan_to_num(M_L, nan=0.0, posinf=0.0, neginf=0.0)
        M_R = numpy.nan_to_num(M_R, nan=0.0, posinf=0.0, neginf=0.0)

        M_bar_sq = 0.5 * (M_L**2 + M_R**2)
        M_bar = numpy.sqrt(numpy.maximum(M_bar_sq, 0.0))

        M_0_p = numpy.minimum(1.0, numpy.maximum(M_bar, M_INF_P))
        fa_p = M_0_p * (2.0 - M_0_p)

        M_0_u = numpy.minimum(1.0, numpy.maximum(M_bar, M_INF_U))
        fa_u = M_0_u * (2.0 - M_0_u)

        alpha = 0.1875 * (-4.0 + 5.0 * fa_p**2)

        M_L_plus = 0.25 * (M_L + 1.0)**2 * (1.0 + 4.0 * beta * (M_L - 1.0)**2)
        M_R_minus = -0.25 * (M_R - 1.0)**2 * (1.0 + 4.0 * beta * (M_R + 1.0)**2)

        # ============================================================
        # Physical acoustic speeds for Mp ONLY
        # ============================================================

        c_L = numpy.sqrt(heat_capacity_ratio * p_L / numpy.maximum(rho_L, 1.0e-12))
        c_R = numpy.sqrt(heat_capacity_ratio * p_R / numpy.maximum(rho_R, 1.0e-12))
        c_half = 0.5 * (c_L + c_R)

        rho_half = 0.5 * (rho_L + rho_R)

        Mp = -(K_p / numpy.maximum(fa_p, 1.0e-12)) * numpy.maximum(1.0 - sigma * M_bar_sq, 0.0) * (p_R - p_L) / numpy.maximum(rho_half * c_half**2, 1.0e-12)

        M = M_L_plus + M_R_minus + Mp

        P_L_plus_coeff = 0.25 * (M_L + 1.0)**2 * (2.0 - M_L + 4.0 * alpha * M_L * (M_L - 1.0)**2)
        P_R_minus_coeff = 0.25 * (M_R - 1.0)**2 * (2.0 + M_R - 4.0 * alpha * M_R * (M_R + 1.0)**2)

        P_L_plus = P_L_plus_coeff * p_L
        P_R_minus = P_R_minus_coeff * p_R

        Pw = -K_u * P_L_plus_coeff * P_R_minus_coeff * (rho_L + rho_R) * fa_u * a_half * (u_R - u_L)

        P_face = P_L_plus + P_R_minus + Pw

        # ============================================================
        # Assemble X1 flux
        # ============================================================

        adv_flux_i = sqrtG_i * (numpy.maximum(0.0, M) * a_L * variables_itf_i[:, :, elem_L, 1, :] + numpy.minimum(0.0, M) * a_R * variables_itf_i[:, :, elem_R, 0, :])

        flux_x1_itf_i[:, :, elem_L, :, 1] = adv_flux_i

        commonPi = sqrtG_i * P_face

        flux_x1_itf_i[idx_rho_u1, :, elem_L, :, 1] += h11 * commonPi
        flux_x1_itf_i[idx_rho_u2, :, elem_L, :, 1] += h12 * commonPi
        flux_x1_itf_i[idx_rho_w, :, elem_L, :, 1] += h13 * commonPi

        flux_x1_itf_i[:, :, elem_R, :, 0] = flux_x1_itf_i[:, :, elem_L, :, 1]

        # ------------------------------------------------------------
        # rho-w advective contribution
        # ------------------------------------------------------------

        w_adv_face_i = adv_flux_i[idx_rho_w, :, :]
        wflux_adv_x1_itf_i[:, elem_L, :, 1] = w_adv_face_i
        wflux_adv_x1_itf_i[:, elem_R, :, 0] = w_adv_face_i

        # ------------------------------------------------------------
        # rho-w pressure contribution
        # ------------------------------------------------------------

        w_pres_face_i = h13 * commonPi
        wflux_pres_x1_itf_i[:, elem_L, :, 1] = w_pres_face_i / numpy.maximum(p_L, 1.0e-12)
        wflux_pres_x1_itf_i[:, elem_R, :, 0] = w_pres_face_i / numpy.maximum(p_R, 1.0e-12)


        # ============================================================
        # X2 DIRECTION
        # ============================================================

        sqrtG_j = metric.sqrtG_itf_j[:, itf, :]
        h21 = metric.H_contra_21_itf_j[:, itf, :]
        h22 = metric.H_contra_22_itf_j[:, itf, :]
        h23 = metric.H_contra_23_itf_j[:, itf, :]

        rho_L = variables_itf_j[idx_rho, :, elem_L, 1, :]
        rho_R = variables_itf_j[idx_rho, :, elem_R, 0, :]

        p_L = pressure_itf_j[:, elem_L, 1, :]
        p_R = pressure_itf_j[:, elem_R, 0, :]

        v_L = u2_itf_j[:, elem_L, 1, :]
        v_R = u2_itf_j[:, elem_R, 0, :]

        a_L = numpy.sqrt(h22 * heat_capacity_ratio * p_L / numpy.maximum(rho_L, 1.0e-12))
        a_R = numpy.sqrt(h22 * heat_capacity_ratio * p_R / numpy.maximum(rho_R, 1.0e-12))
        a_half = 0.5 * (a_L + a_R)

        M_L = v_L / numpy.maximum(a_L, 1.0e-14)
        M_R = v_R / numpy.maximum(a_R, 1.0e-14)

        M_L = numpy.nan_to_num(M_L, nan=0.0, posinf=0.0, neginf=0.0)
        M_R = numpy.nan_to_num(M_R, nan=0.0, posinf=0.0, neginf=0.0)

        M_bar_sq = 0.5 * (M_L**2 + M_R**2)
        M_bar = numpy.sqrt(numpy.maximum(M_bar_sq, 0.0))

        M_0_p = numpy.minimum(1.0, numpy.maximum(M_bar, M_INF_P))
        fa_p = M_0_p * (2.0 - M_0_p)

        M_0_u = numpy.minimum(1.0, numpy.maximum(M_bar, M_INF_U))
        fa_u = M_0_u * (2.0 - M_0_u)

        alpha = 0.1875 * (-4.0 + 5.0 * fa_p**2)

        M_L_plus = 0.25 * (M_L + 1.0)**2 * (1.0 + 4.0 * beta * (M_L - 1.0)**2)
        M_R_minus = -0.25 * (M_R - 1.0)**2 * (1.0 + 4.0 * beta * (M_R + 1.0)**2)

        # ============================================================
        # Physical acoustic speeds for Mp ONLY
        # ============================================================

        c_L = numpy.sqrt(heat_capacity_ratio * p_L / numpy.maximum(rho_L, 1.0e-12))
        c_R = numpy.sqrt(heat_capacity_ratio * p_R / numpy.maximum(rho_R, 1.0e-12))
        c_half = 0.5 * (c_L + c_R)

        rho_half = 0.5 * (rho_L + rho_R)

        Mp = -(K_p / numpy.maximum(fa_p, 1.0e-12)) * numpy.maximum(1.0 - sigma * M_bar_sq, 0.0) * (p_R - p_L) / numpy.maximum(rho_half * c_half**2, 1.0e-12)
        M = M_L_plus + M_R_minus + Mp

        P_L_plus_coeff = 0.25 * (M_L + 1.0)**2 * (2.0 - M_L + 4.0 * alpha * M_L * (M_L - 1.0)**2)
        P_R_minus_coeff = 0.25 * (M_R - 1.0)**2 * (2.0 + M_R - 4.0 * alpha * M_R * (M_R + 1.0)**2)

        P_L_plus = P_L_plus_coeff * p_L
        P_R_minus = P_R_minus_coeff * p_R

        Pw = -K_u * P_L_plus_coeff * P_R_minus_coeff * (rho_L + rho_R) * fa_u * a_half * (v_R - v_L)

        P_face = P_L_plus + P_R_minus + Pw

        # ============================================================
        # Assemble X2 flux
        # ============================================================

        adv_flux_j = sqrtG_j * (numpy.maximum(0.0, M) * a_L * variables_itf_j[:, :, elem_L, 1, :] + numpy.minimum(0.0, M) * a_R * variables_itf_j[:, :, elem_R, 0, :])

        flux_x2_itf_j[:, :, elem_L, 1, :] = adv_flux_j

        commonPj = sqrtG_j * P_face

        flux_x2_itf_j[idx_rho_u1, :, elem_L, 1, :] += h21 * commonPj
        flux_x2_itf_j[idx_rho_u2, :, elem_L, 1, :] += h22 * commonPj
        flux_x2_itf_j[idx_rho_w, :, elem_L, 1, :] += h23 * commonPj

        flux_x2_itf_j[:, :, elem_R, 0, :] = flux_x2_itf_j[:, :, elem_L, 1, :]

        # ------------------------------------------------------------
        # rho-w advective contribution
        # ------------------------------------------------------------

        w_adv_face_j = adv_flux_j[idx_rho_w, :, :]
        wflux_adv_x2_itf_j[:, elem_L, 1, :] = w_adv_face_j
        wflux_adv_x2_itf_j[:, elem_R, 0, :] = w_adv_face_j

        # ------------------------------------------------------------
        # rho-w pressure contribution
        # ------------------------------------------------------------

        w_pres_face_j = h23 * commonPj
        wflux_pres_x2_itf_j[:, elem_L, 1, :] = w_pres_face_j / numpy.maximum(p_L, 1.0e-12)
        wflux_pres_x2_itf_j[:, elem_R, 0, :] = w_pres_face_j / numpy.maximum(p_R, 1.0e-12)


def rusanov_3d_vert(
    variables_itf_k, pressure_itf_k, metric,
    flux_x3_itf_k, wflux_adv_x3_itf_k, wflux_pres_x3_itf_k,
    nb_interfaces_vert, advection_only
):
    """Compute Rusanov fluxes at vertical interfaces."""
    for itf in range(nb_interfaces_vert):
        elem_D = itf
        elem_U = itf + 1

        sqrtG_face = metric.sqrtG_itf_k[itf, :, :]
        h31_face = metric.H_contra_31_itf_k[itf, :, :]
        h32_face = metric.H_contra_32_itf_k[itf, :, :]
        h33_face = metric.H_contra_33_itf_k[itf, :, :]

        rho_D = variables_itf_k[idx_rho, :, elem_D, 1, :]
        rho_U = variables_itf_k[idx_rho, :, elem_U, 0, :]
        w_D = variables_itf_k[idx_rho_w, :, elem_D, 1, :] / rho_D
        w_U = variables_itf_k[idx_rho_w, :, elem_U, 0, :] / rho_U
        p_D = pressure_itf_k[:, elem_D, 1, :]
        p_U = pressure_itf_k[:, elem_U, 0, :]

        if advection_only:
            eig_D = numpy.abs(w_D)
            eig_U = numpy.abs(w_U)
        else:
            eig_D = numpy.abs(w_D) + numpy.sqrt(
                h33_face * heat_capacity_ratio * p_D / numpy.maximum(rho_D, 1e-12)
            )
            eig_U = numpy.abs(w_U) + numpy.sqrt(
                h33_face * heat_capacity_ratio * p_U / numpy.maximum(rho_U, 1e-12)
            )

        eig = numpy.maximum(eig_D, eig_U)

        flux_D = sqrtG_face * w_D * variables_itf_k[:, :, elem_D, 1, :]
        flux_U = sqrtG_face * w_U * variables_itf_k[:, :, elem_U, 0, :]

        wflux_adv_D = flux_D[idx_rho_w].copy()
        wflux_adv_U = flux_U[idx_rho_w].copy()

        flux_D[idx_rho_u1] += sqrtG_face * h31_face * p_D
        flux_D[idx_rho_u2] += sqrtG_face * h32_face * p_D
        flux_D[idx_rho_w] += sqrtG_face * h33_face * p_D

        flux_U[idx_rho_u1] += sqrtG_face * h31_face * p_U
        flux_U[idx_rho_u2] += sqrtG_face * h32_face * p_U
        flux_U[idx_rho_w] += sqrtG_face * h33_face * p_U

        wflux_pres_D = sqrtG_face * h33_face * p_D
        wflux_pres_U = sqrtG_face * h33_face * p_U

        state_jump = (
            variables_itf_k[:, :, elem_U, 0, :]
            - variables_itf_k[:, :, elem_D, 1, :]
        )
        face_flux = 0.5 * (
            flux_D + flux_U - eig * sqrtG_face * state_jump
        )
        flux_x3_itf_k[:, :, elem_D, 1, :] = face_flux
        flux_x3_itf_k[:, :, elem_U, 0, :] = face_flux

        momentum_jump = (
            variables_itf_k[idx_rho_w, :, elem_U, 0, :]
            - variables_itf_k[idx_rho_w, :, elem_D, 1, :]
        )
        w_adv_face = 0.5 * (
            wflux_adv_D + wflux_adv_U - eig * sqrtG_face * momentum_jump
        )
        wflux_adv_x3_itf_k[:, elem_D, 1, :] = w_adv_face
        wflux_adv_x3_itf_k[:, elem_U, 0, :] = w_adv_face

        w_pres_face = 0.5 * (wflux_pres_D + wflux_pres_U)
        wflux_pres_x3_itf_k[:, elem_D, 1, :] = w_pres_face / numpy.maximum(p_D, 1e-12)
        wflux_pres_x3_itf_k[:, elem_U, 0, :] = w_pres_face / numpy.maximum(p_U, 1e-12)


def rusanov_3d_hori(
    variables_itf_i, pressure_itf_i, u1_itf_i,
    variables_itf_j, pressure_itf_j, u2_itf_j,
    metric,
    flux_x1_itf_i, wflux_adv_x1_itf_i, wflux_pres_x1_itf_i,
    flux_x2_itf_j, wflux_adv_x2_itf_j, wflux_pres_x2_itf_j,
    nb_interfaces_hori, advection_only
):
    """Compute Rusanov fluxes at horizontal interfaces."""
    for itf in range(nb_interfaces_hori):
        elem_L = itf
        elem_R = itf + 1

        # X1 direction
        sqrtG_i = metric.sqrtG_itf_i[:, :, itf]
        h11 = metric.H_contra_11_itf_i[:, :, itf]
        h12 = metric.H_contra_12_itf_i[:, :, itf]
        h13 = metric.H_contra_13_itf_i[:, :, itf]

        rho_L = variables_itf_i[idx_rho, :, elem_L, 1, :]
        rho_R = variables_itf_i[idx_rho, :, elem_R, 0, :]
        p_L = pressure_itf_i[:, elem_L, 1, :]
        p_R = pressure_itf_i[:, elem_R, 0, :]
        u_L = u1_itf_i[:, elem_L, 1, :]
        u_R = u1_itf_i[:, elem_R, 0, :]

        if advection_only:
            eig_L = numpy.abs(u_L)
            eig_R = numpy.abs(u_R)
        else:
            eig_L = numpy.abs(u_L) + numpy.sqrt(
                h11 * heat_capacity_ratio * p_L / numpy.maximum(rho_L, 1e-12)
            )
            eig_R = numpy.abs(u_R) + numpy.sqrt(
                h11 * heat_capacity_ratio * p_R / numpy.maximum(rho_R, 1e-12)
            )
        eig = numpy.maximum(eig_L, eig_R)

        flux_L = sqrtG_i * u_L * variables_itf_i[:, :, elem_L, 1, :]
        flux_R = sqrtG_i * u_R * variables_itf_i[:, :, elem_R, 0, :]
        wflux_adv_L = flux_L[idx_rho_w].copy()
        wflux_adv_R = flux_R[idx_rho_w].copy()

        flux_L[idx_rho_u1] += sqrtG_i * h11 * p_L
        flux_L[idx_rho_u2] += sqrtG_i * h12 * p_L
        flux_L[idx_rho_w] += sqrtG_i * h13 * p_L
        flux_R[idx_rho_u1] += sqrtG_i * h11 * p_R
        flux_R[idx_rho_u2] += sqrtG_i * h12 * p_R
        flux_R[idx_rho_w] += sqrtG_i * h13 * p_R

        state_jump = (
            variables_itf_i[:, :, elem_R, 0, :]
            - variables_itf_i[:, :, elem_L, 1, :]
        )
        face_flux_i = 0.5 * (flux_L + flux_R - eig * sqrtG_i * state_jump)
        flux_x1_itf_i[:, :, elem_L, :, 1] = face_flux_i
        flux_x1_itf_i[:, :, elem_R, :, 0] = face_flux_i

        momentum_jump = (
            variables_itf_i[idx_rho_w, :, elem_R, 0, :]
            - variables_itf_i[idx_rho_w, :, elem_L, 1, :]
        )
        w_adv_face_i = 0.5 * (
            wflux_adv_L + wflux_adv_R - eig * sqrtG_i * momentum_jump
        )
        wflux_adv_x1_itf_i[:, elem_L, :, 1] = w_adv_face_i
        wflux_adv_x1_itf_i[:, elem_R, :, 0] = w_adv_face_i

        wflux_pres_L = sqrtG_i * h13 * p_L
        wflux_pres_R = sqrtG_i * h13 * p_R
        w_pres_face_i = 0.5 * (wflux_pres_L + wflux_pres_R)
        wflux_pres_x1_itf_i[:, elem_L, :, 1] = w_pres_face_i / numpy.maximum(p_L, 1e-12)
        wflux_pres_x1_itf_i[:, elem_R, :, 0] = w_pres_face_i / numpy.maximum(p_R, 1e-12)

        # X2 direction
        sqrtG_j = metric.sqrtG_itf_j[:, itf, :]
        h21 = metric.H_contra_21_itf_j[:, itf, :]
        h22 = metric.H_contra_22_itf_j[:, itf, :]
        h23 = metric.H_contra_23_itf_j[:, itf, :]

        rho_L = variables_itf_j[idx_rho, :, elem_L, 1, :]
        rho_R = variables_itf_j[idx_rho, :, elem_R, 0, :]
        p_L = pressure_itf_j[:, elem_L, 1, :]
        p_R = pressure_itf_j[:, elem_R, 0, :]
        v_L = u2_itf_j[:, elem_L, 1, :]
        v_R = u2_itf_j[:, elem_R, 0, :]

        if advection_only:
            eig_L = numpy.abs(v_L)
            eig_R = numpy.abs(v_R)
        else:
            eig_L = numpy.abs(v_L) + numpy.sqrt(
                h22 * heat_capacity_ratio * p_L / numpy.maximum(rho_L, 1e-12)
            )
            eig_R = numpy.abs(v_R) + numpy.sqrt(
                h22 * heat_capacity_ratio * p_R / numpy.maximum(rho_R, 1e-12)
            )
        eig = numpy.maximum(eig_L, eig_R)

        flux_L = sqrtG_j * v_L * variables_itf_j[:, :, elem_L, 1, :]
        flux_R = sqrtG_j * v_R * variables_itf_j[:, :, elem_R, 0, :]
        wflux_adv_L = flux_L[idx_rho_w].copy()
        wflux_adv_R = flux_R[idx_rho_w].copy()

        flux_L[idx_rho_u1] += sqrtG_j * h21 * p_L
        flux_L[idx_rho_u2] += sqrtG_j * h22 * p_L
        flux_L[idx_rho_w] += sqrtG_j * h23 * p_L
        flux_R[idx_rho_u1] += sqrtG_j * h21 * p_R
        flux_R[idx_rho_u2] += sqrtG_j * h22 * p_R
        flux_R[idx_rho_w] += sqrtG_j * h23 * p_R

        state_jump = (
            variables_itf_j[:, :, elem_R, 0, :]
            - variables_itf_j[:, :, elem_L, 1, :]
        )
        face_flux_j = 0.5 * (flux_L + flux_R - eig * sqrtG_j * state_jump)
        flux_x2_itf_j[:, :, elem_L, 1, :] = face_flux_j
        flux_x2_itf_j[:, :, elem_R, 0, :] = face_flux_j

        momentum_jump = (
            variables_itf_j[idx_rho_w, :, elem_R, 0, :]
            - variables_itf_j[idx_rho_w, :, elem_L, 1, :]
        )
        w_adv_face_j = 0.5 * (
            wflux_adv_L + wflux_adv_R - eig * sqrtG_j * momentum_jump
        )
        wflux_adv_x2_itf_j[:, elem_L, 1, :] = w_adv_face_j
        wflux_adv_x2_itf_j[:, elem_R, 0, :] = w_adv_face_j

        wflux_pres_L = sqrtG_j * h23 * p_L
        wflux_pres_R = sqrtG_j * h23 * p_R
        w_pres_face_j = 0.5 * (wflux_pres_L + wflux_pres_R)
        wflux_pres_x2_itf_j[:, elem_L, 1, :] = w_pres_face_j / numpy.maximum(p_L, 1e-12)
        wflux_pres_x2_itf_j[:, elem_R, 0, :] = w_pres_face_j / numpy.maximum(p_R, 1e-12)


#@profile
def rhs_euler_core (Q: numpy.ndarray, geom: CubedSphere, mtrx: DFROperators, metric: Metric3DTopo, ptopo: DistributedWorld,
               nbsolpts: int, nb_elements_hori: int, nb_elements_vert: int, case_number: int):
   '''Evaluate the right-hand side of the three-dimensional Euler equations.

   This function evaluates RHS of the Euler equations using the four-demsional tensor formulation (see Charron 2014), returning
   an array consisting of the time-derivative of the conserved variables (ρ,ρu,ρv,ρw,ρθ).  A "curried" version of this function,
   with non-Q parameters predefined, should be passed to the time-stepping routine to use as a RHS black-box.  Since some of the
   time-stepping routines perform a Jacobian evaluation via complex derivative, this function should also be safe with respect to
   complex-valued inputs inside Q.

   Note that this function includes MPI communication for inter-process boundary interactions, so it must be called collectively.

   Parameters
   ----------
   Q : numpy.ndarray
      Input array of the current model state, indexed as (var,k,j,i)
   geom : CubedSphere
      Geometry definition, containing parameters relating to the spherical coordinate system
   mtrx : DFR_operators
      Contains matrix operators for the DFR discretization, notably boundary extrapolation and
      local (partial) derivatives
   metric : Metric
      Contains the various metric terms associated with the tensor formulation, notably including the
      scalar √g, the spatial metric h, and the Christoffel symbols
   ptopo : Distributed_World
      Wraps the information and communication functions necessary for MPI distribution
   nbsolpts : int
      Number of interior nodal points per element.  A 3D element will contain nbsolpts**3 internal points.
   nb_elements_hori : int
      Number of elements in x/y on each panel of the cubed sphere
   nb_elements_vert : int
      Number of elements in the vertical
   case_number : int
      DCMIP case number, used to selectively enable or disable parts of the Euler equations to accomplish
      specialized tests like advection-only

   Returns:
   --------
   rhs : numpy.ndarray
      Output of right-hand-side terms of Euler equations
   '''
   
   riemann_solver = _validate_riemann_solver(RIEMANN_SOLVER)

   # print(metric.H_contra_13.max())
   type_vec = Q.dtype #  Output/processing type -- may be complex
   nb_equations = Q.shape[0] # Number of constituent Euler equations.  Probably 6.
   nb_interfaces_hori = nb_elements_hori + 1 # Number of element interfaces per horizontal dimension
   nb_interfaces_vert = nb_elements_vert + 1 # Number of element interfaces in the vertical dimension
   nb_pts_hori = nb_elements_hori * nbsolpts # Total number of solution points per horizontal dimension
   nb_vertical_levels = nb_elements_vert * nbsolpts # Total number of solution points in the vertical dimension

   # Create new arrays for each component of T^μν_:ν, plus one more for the final right hand side
   df1_dx1, df2_dx2, df3_dx3, rhs = [numpy.empty_like(Q, dtype=type_vec) for _ in range(4)]

   # Array for forcing: Coriolis terms, metric corrections from the curvilinear coordinate, and gravity
   forcing = numpy.zeros_like(Q, dtype=type_vec)

   # Array to extrapolate variables and fluxes to the boundaries along x (i)
   variables_itf_i = numpy.ones((nb_equations, nb_vertical_levels, nb_elements_hori + 2, 2, nb_pts_hori), dtype=type_vec) # Initialized to one in the halo to avoid division by zero later
   # Note that flux_x1_itf_i has a different shape than variables_itf_i
   flux_x1_itf_i   = numpy.empty((nb_equations, nb_vertical_levels, nb_elements_hori + 2, nb_pts_hori, 2), dtype=type_vec)

   # Extrapolation arrays along y (j)
   variables_itf_j = numpy.ones((nb_equations, nb_vertical_levels, nb_elements_hori + 2, 2, nb_pts_hori), dtype=type_vec) # Initialized to one in the halo to avoid division by zero later
   flux_x2_itf_j   = numpy.empty((nb_equations, nb_vertical_levels, nb_elements_hori + 2, 2, nb_pts_hori), dtype=type_vec)

   # Extrapolation arrays along z (k), note dimensions of (6, nj, nk+2, 2, ni)
   variables_itf_k = numpy.empty((nb_equations, nb_pts_hori, nb_elements_vert + 2, 2, nb_pts_hori), dtype=type_vec)
   flux_x3_itf_k   = numpy.empty((nb_equations, nb_pts_hori, nb_elements_vert + 2, 2, nb_pts_hori), dtype=type_vec)

   # Special arrays for calculation of (ρw) flux
   wflux_adv_x1_itf_i = numpy.zeros_like(flux_x1_itf_i[0,:])
   wflux_pres_x1_itf_i = numpy.zeros_like(wflux_adv_x1_itf_i)
   wflux_adv_x2_itf_j = numpy.zeros_like(flux_x2_itf_j[0,:])
   wflux_pres_x2_itf_j = numpy.zeros_like(wflux_adv_x2_itf_j)
   wflux_adv_x3_itf_k = numpy.zeros_like(flux_x3_itf_k[0,:])
   wflux_pres_x3_itf_k = numpy.zeros_like(flux_x3_itf_k[0,:])

   # Flag for advection-only processing, with DCMIP test cases 11 and 12
   advection_only = case_number < 13

   # Offset due to the halo
   offset = 1

   # Interpolate to the element interface
   # for elem in range(nb_elements_hori):
   #    # This loop performs extrapolation to element boundaries through the mtrix.extrap_* operator (matrix multiplication).
   #    # Thanks to numpy's broadcasting, each iteration of this loop extrapolates an entire row/column of elements at once,
   #    # operating on all variables simultaneously

   #    # Index of the 'live' interior elements inside the Q array, to be extrapolated
   #    epais = elem * nbsolpts + numpy.arange(nbsolpts)
   #    # Position in the output interface array for writing.  'pos' 1 corresponds to the west/southmost element, with
   #    # 'pos' 0 (and nb_elements_hori+1) reserved for exchanges from neighbouring panels
   #    pos   = elem + offset

   #    # --- Direction x1
   #    # The implied matrix multiplication here sees a [numvar, nk] array of matrices, each 
   #    # of size [nj, nbsolpoints], and the extrapolation is performed via right multiplication.
   #    # (Note C-ordering of indices; in fortran or matlab the indices would be reversed)
   #    variables_itf_i[:, :, pos, 0, :] = Q[:, :, :, epais] @ mtrx.extrap_west
   #    variables_itf_i[:, :, pos, 1, :] = Q[:, :, :, epais] @ mtrx.extrap_east

   #    # --- Direction x2
   #    # The matrix multiplication here sees a [numvar, nk] array of matrices, each of size
   #    # [nbsolpoints, ni], and the extrapolation is performed by left multiplication
   #    variables_itf_j[:, :, pos, 0, :] = mtrx.extrap_south @ Q[:, :, epais, :]
   #    variables_itf_j[:, :, pos, 1, :] = mtrx.extrap_north @ Q[:, :, epais, :]

   variables_itf_i[:,:,1:-1,:,:] = mtrx.extrapolate_i(Q,geom).transpose((0,1,3,4,2))
   variables_itf_j[:,:,1:-1,:,:] = mtrx.extrapolate_j(Q,geom)

   # Scaled variables for separate reconstruction
   logrho = numpy.log(Q[idx_rho,:])
   logrhotheta = numpy.log(Q[idx_rho_theta,:])

   variables_itf_i[idx_rho,:,1:-1,:,:] = numpy.exp(mtrx.extrapolate_i(logrho,geom)).transpose((0,2,3,1))
   variables_itf_j[idx_rho,:,1:-1,:,:] = numpy.exp(mtrx.extrapolate_j(logrho,geom))

   variables_itf_i[idx_rho_theta,:,1:-1,:,:] = numpy.exp(mtrx.extrapolate_i(logrhotheta,geom)).transpose((0,2,3,1))
   variables_itf_j[idx_rho_theta,:,1:-1,:,:] = numpy.exp(mtrx.extrapolate_j(logrhotheta,geom))

   # Transfer boundary values to neighbouring proessors/panels, including conversion of vector quantities
   # to the recipient's local coordinate system

   # Initiate transfers
   all_request = ptopo.xchange_Euler_interfaces(geom, variables_itf_i, variables_itf_j, blocking=False)

   # Unpack dynamical variables, each to arrays of size [nk,nj,ni]
   rho = Q[idx_rho]
   u1  = Q[idx_rho_u1] / rho
   u2  = Q[idx_rho_u2] / rho
   w   = Q[idx_rho_w]  / rho # TODO : u3

   # Compute the fluxes (equation 3 of Charron & Gaudreault 2021, LHS)

   # Compute the advective fluxes ...
   flux_x1 = metric.sqrtG * u1 * Q
   flux_x2 = metric.sqrtG * u2 * Q
   flux_x3 = metric.sqrtG * w  * Q

   wflux_adv_x1 = metric.sqrtG * u1 * Q[idx_rho_w,:]
   wflux_adv_x2 = metric.sqrtG * u2 * Q[idx_rho_w,:]
   wflux_adv_x3 = metric.sqrtG * w  * Q[idx_rho_w,:]

   # ... and add the pressure component
   # Performance note: exp(log) is measuably faster than ** (pow)
   pressure = p0 * numpy.exp((cpd/cvd) * numpy.log((Rd/p0)*Q[idx_rho_theta]))
   #pressure = Rd * Q[idx_rho_theta]

   wflux_pres_x1 = numpy.zeros(metric.sqrtG.shape,dtype=type_vec)
   wflux_pres_x2 = numpy.zeros(metric.sqrtG.shape,dtype=type_vec)
   wflux_pres_x3 = numpy.zeros(metric.sqrtG.shape,dtype=type_vec)

   flux_x1[idx_rho_u1] += metric.sqrtG * metric.H_contra_11 * pressure
   flux_x1[idx_rho_u2] += metric.sqrtG * metric.H_contra_12 * pressure
   flux_x1[idx_rho_w]  += metric.sqrtG * metric.H_contra_13 * pressure

   wflux_pres_x1[:] = metric.sqrtG * metric.H_contra_13 # times pressure

   flux_x2[idx_rho_u1] += metric.sqrtG * metric.H_contra_21 * pressure
   flux_x2[idx_rho_u2] += metric.sqrtG * metric.H_contra_22 * pressure
   flux_x2[idx_rho_w]  += metric.sqrtG * metric.H_contra_23 * pressure

   wflux_pres_x2[:] = metric.sqrtG * metric.H_contra_23 # times pressure

   flux_x3[idx_rho_u1] += metric.sqrtG * metric.H_contra_31 * pressure
   flux_x3[idx_rho_u2] += metric.sqrtG * metric.H_contra_32 * pressure
   flux_x3[idx_rho_w]  += metric.sqrtG * metric.H_contra_33 * pressure

   wflux_pres_x3[:] = metric.sqrtG * metric.H_contra_33 # times pressure

   # if (ptopo.rank == 0): print('√g: %e, H^33: %e' % (metric.sqrtG[0,0],metric.H_contra_33[0,0]))

   # Interior contribution to the derivatives, corrections for the boundaries will be added later
   # The "interior contribution" here is evaluated as if the fluxes at the element boundaries are
   # zero.
   # for elem in range(nb_elements_hori):
   #    epais = elem * nbsolpts + numpy.arange(nbsolpts)

   #    # --- Direction x1
   #    df1_dx1[:, :, :, epais] = flux_x1[:, :, :, epais] @ mtrx.diff_solpt_tr

   #    # --- Direction x2
   #    df2_dx2[:, :, epais, :] = mtrx.diff_solpt @ flux_x2[:, :, epais, :]

   # --- Direction x3

   # Important notice : all the vertical stuff should be done before the synchronization of the horizontal communications.
   # Since there is no communication step in the vertical, we can compute the boundary correction first

   # Extrapolate to top/bottom for each element
   # for slab in range(nb_pts_hori):
   #    for elem in range(nb_elements_vert):
   #       epais = elem * nbsolpts + numpy.arange(nbsolpts)
   #       pos = elem + offset

   #       # Extrapolate by left multiplication.  Note that this assignment also permutes the indices, going from
   #       # (var,nk,nj,ni) to (var,nj,nk,top/bot,ni)
   #       variables_itf_k[:, slab, pos, 0, :] = mtrx.extrap_down @ Q[:, epais, slab, :]
   #       variables_itf_k[:, slab, pos, 1, :] = mtrx.extrap_up   @ Q[:, epais, slab, :]

   variables_itf_k[:,:,1:-1,:,:] = mtrx.extrapolate_k(Q,geom).transpose((0,3,1,2,4))

   variables_itf_k[idx_rho,:,1:-1,:,:] = numpy.exp(mtrx.extrapolate_k(logrho,geom).transpose((2,0,1,3)))
   variables_itf_k[idx_rho_theta,:,1:-1,:,:] = numpy.exp(mtrx.extrapolate_k(logrhotheta,geom).transpose((2,0,1,3)))

   # For consistency at the surface and top boundaries, treat the extrapolation as continuous.  That is,
   # the "top" of the ground is equal to the "bottom" of the atmosphere, and the "bottom" of the model top
   # is equal to the "top" of the atmosphere.
   variables_itf_k[:, :, 0, 1, :] = variables_itf_k[:, :, 1, 0, :]
   variables_itf_k[:, :, 0, 0, :] = variables_itf_k[:, :, 0, 1, :] # Unused?
   variables_itf_k[:, :, -1, 0, :] = variables_itf_k[:, :, -2, 1, :]
   variables_itf_k[:, :, -1, 1, :] = variables_itf_k[:, :, -1, 0, :] # Unused?

   # Enforce no-flow through the top and botttom by imposing odd symmetry on ρw
   variables_itf_k[idx_rho_w,:,0,1,:] = -variables_itf_k[idx_rho_w,:,1,0,:]
   variables_itf_k[idx_rho_w,:,-1,0,:] = -variables_itf_k[idx_rho_w,:,-2,1,:]

   # Evaluate pressure at the vertical element interfaces based on ρθ.
   pressure_itf_k = p0 * numpy.exp((cpd/cvd)*numpy.log(variables_itf_k[idx_rho_theta] * (Rd / p0)))

   # Take w ← (wρ)/ ρ at the vertical interfaces
   w_itf_k = variables_itf_k[idx_rho_w] / variables_itf_k[idx_rho]

   # Select the vertical interface flux.
   if riemann_solver == "rusanov":
      rusanov_3d_vert(
         variables_itf_k, pressure_itf_k, metric,
         flux_x3_itf_k, wflux_adv_x3_itf_k, wflux_pres_x3_itf_k,
         nb_interfaces_vert, advection_only
      )
   else:  # ausmplusup
      ausm_3d_vert(
         variables_itf_k, pressure_itf_k, metric,
         flux_x3_itf_k, wflux_adv_x3_itf_k, wflux_pres_x3_itf_k,
         nb_interfaces_vert
      )

   
   # Finish transfers
   all_request.wait()

   # sys.exit(1)

   # Define u, v at the interface by dividing momentum and density
   u1_itf_i = variables_itf_i[idx_rho_u1] / variables_itf_i[idx_rho]
   u2_itf_j = variables_itf_j[idx_rho_u2] / variables_itf_j[idx_rho]

   # Evaluate pressure at the lateral interfaces
   pressure_itf_i = p0 * numpy.exp((cpd/cvd) * numpy.log(variables_itf_i[idx_rho_theta] * (Rd / p0)))
   pressure_itf_j = p0 * numpy.exp((cpd/cvd) * numpy.log(variables_itf_j[idx_rho_theta] * (Rd / p0)))

   # Select the horizontal interface flux.
   if riemann_solver == "rusanov":
      rusanov_3d_hori(
         variables_itf_i, pressure_itf_i, u1_itf_i,
         variables_itf_j, pressure_itf_j, u2_itf_j,
         metric,
         flux_x1_itf_i, wflux_adv_x1_itf_i, wflux_pres_x1_itf_i,
         flux_x2_itf_j, wflux_adv_x2_itf_j, wflux_pres_x2_itf_j,
         nb_interfaces_hori, advection_only
      )
   else:  # ausmplusup
      ausm_3d_hori_ausmplusup(
         variables_itf_i, pressure_itf_i, u1_itf_i,
         variables_itf_j, pressure_itf_j, u2_itf_j,
         metric,
         flux_x1_itf_i, wflux_adv_x1_itf_i, wflux_pres_x1_itf_i,
         flux_x2_itf_j, wflux_adv_x2_itf_j, wflux_pres_x2_itf_j,
         nb_interfaces_hori, idx_rho, idx_rho_u1, idx_rho_u2,
         idx_rho_w, heat_capacity_ratio
      )
   # # Add corrections to the derivatives
   # for elem in range(nb_elements_hori):
   #    epais = elem * nbsolpts + numpy.arange(nbsolpts)

   #    # --- Direction x1

   #    df1_dx1[:, :, :, epais] += flux_x1_itf_i[:, :, elem+offset, :, :] @ mtrx.correction_tr

   #    # --- Direction x2

   #    df2_dx2[:, :, epais, :] += mtrx.correction @ flux_x2_itf_j[:, :, elem+offset, :, :]

   # Perform flux derivatives

   flux_x1_bdy = flux_x1_itf_i.transpose((0,1,3,2,4))[:,:,:,1:-1,:].copy()
   df1_dx1 = mtrx.comma_i(flux_x1,flux_x1_bdy,geom)
   flux_x2_bdy = flux_x2_itf_j[:,:,1:-1,:,:].copy()
   df2_dx2 = mtrx.comma_j(flux_x2,flux_x2_bdy,geom)
   flux_x3_bdy = flux_x3_itf_k[:,:,1:-1,:,:].transpose(0,2,3,1,4).copy()
   df3_dx3[:,:,:,:] = mtrx.comma_k(flux_x3,flux_x3_bdy,geom)

   logp_int = numpy.log(pressure)

   pressure_bdy_i = pressure_itf_i[:,1:-1,:,:].transpose((0,3,1,2)).copy()
   pressure_bdy_j = pressure_itf_j[:,1:-1,:,:].copy()
   pressure_bdy_k = pressure_itf_k[:,1:-1,:,:].transpose(1,2,0,3).copy()

   logp_bdy_i = numpy.log(pressure_bdy_i)
   logp_bdy_j = numpy.log(pressure_bdy_j)
   logp_bdy_k = numpy.log(pressure_bdy_k)

   wflux_adv_x1_bdy_i = wflux_adv_x1_itf_i.transpose((0,2,1,3))[:,:,1:-1,:].copy()
   wflux_pres_x1_bdy_i = wflux_pres_x1_itf_i.transpose((0,2,1,3))[:,:,1:-1,:].copy()

   wflux_adv_x2_bdy_j = wflux_adv_x2_itf_j[:,1:-1,:,:].copy()
   wflux_pres_x2_bdy_j = wflux_pres_x2_itf_j[:,1:-1,:,:].copy()

   wflux_adv_x3_bdy_k = wflux_adv_x3_itf_k[:,1:-1,:,:].transpose(1,2,0,3).copy()
   wflux_pres_x3_bdy_k = wflux_pres_x3_itf_k[:,1:-1,:,:].transpose(1,2,0,3).copy()

   # dFw/dx = d(adv)/dx + d(pres*metric)/dx = d(adv)/dx + pres*(d(metric)/dx + metric*d(logp)/dx)
   w_df1_dx1_adv = mtrx.comma_i(wflux_adv_x1,wflux_adv_x1_bdy_i,geom)
   w_df1_dx1_presa = pressure*mtrx.comma_i(wflux_pres_x1,wflux_pres_x1_bdy_i,geom)
   w_df1_dx1_presb = pressure*wflux_pres_x1*mtrx.comma_i(logp_int,logp_bdy_i,geom)
   w_df1_dx1 = w_df1_dx1_adv + w_df1_dx1_presa + w_df1_dx1_presb

   # dFw/dy = d(adv)/dy + d(pres*metric)/dy = d(adv)/dy + pres*(d(metric)/dy + metric*d(logp)/dy)
   w_df2_dx2_adv = mtrx.comma_j(wflux_adv_x2,wflux_adv_x2_bdy_j,geom)
   w_df2_dx2_presa = pressure*mtrx.comma_j(wflux_pres_x2,wflux_pres_x2_bdy_j,geom)
   w_df2_dx2_presb = pressure*wflux_pres_x2*mtrx.comma_j(logp_int,logp_bdy_j,geom)
   w_df2_dx2 = w_df2_dx2_adv + w_df2_dx2_presa + w_df2_dx2_presb


   # dFw/dz = d(adv)/dz + d(pres*metric)/dz = d(adv)/dz + pres*(d(metric)/dz + metric*d(logp)/dz)
   w_df3_dx3c = mtrx.comma_k(wflux_adv_x3,wflux_adv_x3_bdy_k,geom) 
   w_df3_dx3a = pressure*mtrx.comma_k(wflux_pres_x3,wflux_pres_x3_bdy_k,geom) 
   w_df3_dx3b = pressure*wflux_pres_x3*mtrx.comma_k(logp_int,logp_bdy_k,geom)
   w_df3_dx3 = w_df3_dx3a + w_df3_dx3b + w_df3_dx3c


   # Add coriolis, metric terms and other forcings
   forcing[idx_rho,:,:,:] = 0.0

   # TODO: could be simplified
   #pressure[:] = 0
   forcing[idx_rho_u1] = 2.0 * ( metric.christoffel_1_01 * rho * u1 + metric.christoffel_1_02 * rho * u2 + metric.christoffel_1_03 * rho * w) \
         +       metric.christoffel_1_11 * (rho * u1 * u1 + metric.H_contra_11*pressure) \
         + 2.0 * metric.christoffel_1_12 * (rho * u1 * u2 + metric.H_contra_12*pressure) \
         + 2.0 * metric.christoffel_1_13 * (rho * u1 * w  + metric.H_contra_13*pressure) \
         +       metric.christoffel_1_22 * (rho * u2 * u2 + metric.H_contra_22*pressure) \
         + 2.0 * metric.christoffel_1_23 * (rho * u2 * w  + metric.H_contra_23*pressure) \
         +       metric.christoffel_1_33 * (rho * w * w   + metric.H_contra_33*pressure)

   forcing[idx_rho_u2] = 2.0 * (metric.christoffel_2_01 * rho * u1 + metric.christoffel_2_02 * rho * u2 + metric.christoffel_2_03 * rho * w) \
         +       metric.christoffel_2_11 * (rho * u1 * u1 + metric.H_contra_11*pressure) \
         + 2.0 * metric.christoffel_2_12 * (rho * u1 * u2 + metric.H_contra_12*pressure) \
         + 2.0 * metric.christoffel_2_13 * (rho * u1 * w  + metric.H_contra_13*pressure) \
         +       metric.christoffel_2_22 * (rho * u2 * u2 + metric.H_contra_22*pressure) \
         + 2.0 * metric.christoffel_2_23 * (rho * u2 * w  + metric.H_contra_23*pressure) \
         +       metric.christoffel_2_33 * (rho * w * w   + metric.H_contra_33*pressure)

   # Note: the gρ term here is written with filter_k, which removes the highest vertical
   # wavenumber.  This appears to be required for stability in the absence of a hyperviscosity
   # filter.  Otherwise, the vertically variable pressure (proportional to exp(-z)) times
   # a grid-scale oscillation in ρ causes aliasing and instability.  We may need to examine
   # a more full de-aliasing procedure, or perhaps separate p/ρ into hydrostatic components 
   # (cancelled analytically) and nonhydrostatic components.
   forcing[idx_rho_w] = 2.0 * (metric.christoffel_3_01 * rho * u1 + metric.christoffel_3_02 * rho * u2 + metric.christoffel_3_03 * rho * w) \
         +       metric.christoffel_3_11 * (rho * u1 * u1 + metric.H_contra_11*pressure) \
         + 2.0 * metric.christoffel_3_12 * (rho * u1 * u2 + metric.H_contra_12*pressure) \
         + 2.0 * metric.christoffel_3_13 * (rho * u1 * w  + metric.H_contra_13*pressure) \
         +       metric.christoffel_3_22 * (rho * u2 * u2 + metric.H_contra_22*pressure) \
         + 2.0 * metric.christoffel_3_23 * (rho * u2 * w  + metric.H_contra_23*pressure) \
         +       metric.christoffel_3_33 * (rho * w * w   + metric.H_contra_33*pressure) \
         + metric.inv_dzdeta * gravity * mtrx.filter_k(rho, geom)
         # + metric.inv_dzdeta * gravity * metric.inv_sqrtG * mtrx.filter_k(metric.sqrtG*rho, geom)
         # + (metric.inv_dzdeta * rho * gravity)
         #+ metric.inv_dzdeta * gravity * numpy.exp(mtrx.filter_k(logrho, geom))



   forcing[idx_rho_theta] = 0.0

   # DCMIP cases 2-1 and 2-2 involve rayleigh damping
   if case_number == 21:
      # dcmip_schar_damping modifies the 'forcing' variable to apply the requried Rayleigh damping
      dcmip_schar_damping(forcing, rho, u1, u2, w, metric, geom, shear=False)
   elif case_number == 22:
      dcmip_schar_damping(forcing, rho, u1, u2, w, metric, geom, shear=True)
   # elif case_number == 41:
   #    dcmip_damping(forcing, rho, u1, u2, w, metric, geom, shear=False)


   # Assemble the right-hand sides
   rhs = - metric.inv_sqrtG * ( df1_dx1 + df2_dx2 + df3_dx3 ) - forcing
   rhs[idx_rho_w,:] = - metric.inv_sqrtG * ( w_df1_dx1 + \
                                             w_df2_dx2 + \
                                             w_df3_dx3 ) - forcing[idx_rho_w,:]

   # For pure advection problems, we do not update the dynamical variables
   if advection_only:
      rhs[idx_rho]       = 0.0
      rhs[idx_rho_u1]    = 0.0
      rhs[idx_rho_u2]    = 0.0
      rhs[idx_rho_w]     = 0.0
      rhs[idx_rho_theta] = 0.0
   return rhs

def build_dcmip31_reference_state( geom, metric, mtrx, param, fields_shape, dtype=numpy.float64):
    """
    Build the unperturbed DCMIP-31 reference state in conserved variables.

    Q_ref = [rho, rho*u1, rho*u2, rho*w, rho*theta, ...]
    """

    rho_ref, u1_ref, u2_ref, w_ref, theta_ref = dcmip_gravity_wave( geom, metric, mtrx, param, perturb=False)

    Q_ref = numpy.zeros(fields_shape, dtype=dtype)

    Q_ref[idx_rho] = rho_ref
    Q_ref[idx_rho_u1] = rho_ref * u1_ref
    Q_ref[idx_rho_u2] = rho_ref * u2_ref
    Q_ref[idx_rho_w] = rho_ref * w_ref
    Q_ref[idx_rho_theta] = rho_ref * theta_ref

    return Q_ref

def build_dcmip20_reference_state( geom, metric, mtrx, param, fields_shape, dtype=numpy.float64,):
    
    rho_ref, u1_ref, u2_ref, w_ref, theta_ref = dcmip_steady_state_mountain(geom, metric, mtrx, param, apply_topography=False,)

    Q_ref = numpy.zeros(fields_shape, dtype=dtype,)

    Q_ref[idx_rho] = rho_ref
    Q_ref[idx_rho_u1] = (rho_ref * u1_ref)
    Q_ref[idx_rho_u2] = (rho_ref * u2_ref)
    Q_ref[idx_rho_w] = (rho_ref * w_ref)
    Q_ref[idx_rho_theta] = (rho_ref * theta_ref)

    return Q_ref
