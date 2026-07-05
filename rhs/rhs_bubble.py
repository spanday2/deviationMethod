import numpy

from common.definitions import idx_2d_rho, idx_2d_rho_u, idx_2d_rho_w, idx_2d_rho_theta,  \
                               p0, Rd, cpd, cvd, heat_capacity_ratio, gravity


def ausm_plus_flux(
    U_L, U_R,                 # (neq, N) left/right conserved states at the interface
    p_L, p_R,                 # (N,) pressures on L/R
    gamma,                    # heat_capacity_ratio
    *,
    idx_rho, idx_u, idx_w, idx_theta,
    normal="z",               # "z" for vertical faces (normal=w), "x" for horizontal (normal=u)
    p_base_L=None, p_base_R=None,
    eps=1e-14):
    """
    Vectorized AUSM+ (simplified, matching your current formula).
    Returns interfacial flux F_int with shape (neq, N).

    U = [rho, rho*u, rho*w, rho*E] with indices passed in.
    """
    neq, N = U_L.shape
    out_dtype = U_L.dtype

    rho_L = U_L[idx_rho]
    rho_R = U_R[idx_rho]

    # Normal velocity component (u or w) on each side
    if normal == "z":
        mom_idx = idx_w
        vL = U_L[idx_w] / rho_L
        vR = U_R[idx_w] / rho_R
    elif normal == "x":
        mom_idx = idx_u
        vL = U_L[idx_u] / rho_L
        vR = U_R[idx_u] / rho_R

    # Speed of sound on each side
    a_L = numpy.sqrt(numpy.maximum(gamma * p_L / rho_L, 0.0))
    a_R = numpy.sqrt(numpy.maximum(gamma * p_R / rho_R, 0.0))

    # Mach numbers
    M_L = vL / a_L
    M_R = vR / a_R

    # Your current AUSM+ face Mach blending
    M_face = 0.25 * ((M_L + 1.0)**2 - (M_R - 1.0)**2)

    # Mass flux split
    m_plus  = numpy.maximum(0.0, M_face) * a_L
    m_minus = numpy.minimum(0.0, M_face) * a_R

    # Convective part (broadcast over equations)
    F = (U_L * m_plus) + (U_R * m_minus)

    # Pressure part to normal momentum (your current formula)
    F[mom_idx] += 0.5 * ((1.0 + M_L) * p_L + (1.0 - M_R) * p_R)

    return F.astype(out_dtype, copy=False)
 

def ausm_plus_up_flux(
    U_L, U_R,                 # (neq, N)
    p_L, p_R,                 # (N,)
    gamma,                    # heat_capacity_ratio
    *,
    idx_rho, idx_u, idx_w, idx_theta,
    normal="z",
    p_base_L=None,
    p_base_R=None,
    eps_scalar_diss=0.005,    # small scalar diffusion for rho and rho*theta
):
    """
    AUSM+up interfacial flux for conserved state:
        [rho, rho*u, rho*w, rho*theta]

    Includes:
      1. AUSM+up pressure and velocity diffusion.
      2. Optional hydrostatic-background pressure removal in the K_p term.
      3. Small Rusanov-like scalar diffusion only for rho and rho*theta.

    eps_scalar_diss:
      0.0   -> no extra scalar diffusion
      0.001 -> very weak
      0.005 -> recommended first test
      0.01  -> stronger
    """

    neq, N = U_L.shape
    dtype = U_L.dtype

    rho_L = U_L[idx_rho]
    rho_R = U_R[idx_rho]

    # ------------------------------------------------------------
    # Normal and tangential velocities
    # ------------------------------------------------------------
    if normal == "z":
        mom_n = idx_w

        vL = U_L[idx_w] / rho_L   # normal velocity w_L
        vR = U_R[idx_w] / rho_R   # normal velocity w_R

        uL = U_L[idx_u] / rho_L   # tangential velocity
        uR = U_R[idx_u] / rho_R

    elif normal == "x":
        mom_n = idx_u

        vL = U_L[idx_u] / rho_L   # normal velocity u_L
        vR = U_R[idx_u] / rho_R   # normal velocity u_R

        wL = U_L[idx_w] / rho_L   # tangential velocity
        wR = U_R[idx_w] / rho_R

    else:
        raise ValueError("normal must be 'z' or 'x'")

    # ------------------------------------------------------------
    # Sound speed
    # ------------------------------------------------------------
    a_L = numpy.sqrt(gamma * p_L / rho_L)
    a_R = numpy.sqrt(gamma * p_R / rho_R)

    a_half = 0.5 * (a_L + a_R)
    a_half_sq = a_half * a_half

    # ------------------------------------------------------------
    # AUSM+up constants
    # ------------------------------------------------------------
    sigma = 1.0
    K_p = 0.25
    K_u = 0.75
    beta = 1.0 / 8.0

    M_inf_u = 1.0e-4
    M_inf_p = 1.0

    # ------------------------------------------------------------
    # Mach numbers
    # ------------------------------------------------------------
    M_L = vL / a_half
    M_R = vR / a_half

    Mbar_sq = (vL**2 + vR**2) / (2.0 * a_half_sq)

    # ------------------------------------------------------------
    # Pressure-diffusion scaling
    # ------------------------------------------------------------
    Mo_p_sq = numpy.minimum(1.0, numpy.maximum(Mbar_sq, M_inf_p**2))
    Mo_p = numpy.sqrt(Mo_p_sq)
    fa_p = Mo_p * (2.0 - Mo_p)

    # ------------------------------------------------------------
    # Convective Mach split
    # ------------------------------------------------------------
    Mplus = (
        0.25 * (M_L + 1.0)**2
        * (1.0 + 16.0 * beta * 0.25 * (M_L - 1.0)**2)
    )

    Mminus = (
        -0.25 * (M_R - 1.0)**2
        * (1.0 + 16.0 * beta * 0.25 * (M_R + 1.0)**2)
    )

    rho_half = 0.5 * (rho_L + rho_R)

    # ------------------------------------------------------------
    # Hydrostatic-background pressure removal for K_p term
    # ------------------------------------------------------------
    if p_base_L is not None and p_base_R is not None:
        dp_for_Kp = (p_R - p_L) - (p_base_R - p_base_L)
    else:
        dp_for_Kp = p_R - p_L

    M_half = (
        Mplus + Mminus
        - K_p * (1.0 / fa_p)
        * numpy.maximum(1.0 - sigma * Mbar_sq, 0.0)
        * (dp_for_Kp / (rho_half * a_half_sq))
    )

    # ------------------------------------------------------------
    # Mass flux
    # ------------------------------------------------------------
    up_L = M_half > 0.0
    mdothalf = a_half * M_half * numpy.where(up_L, rho_L, rho_R)

    # ------------------------------------------------------------
    # Pressure flux split
    # ------------------------------------------------------------
    Mo_u_sq = numpy.minimum(1.0, numpy.maximum(Mbar_sq, M_inf_u**2))
    Mo_u = numpy.sqrt(Mo_u_sq)
    fa_u = Mo_u * (2.0 - Mo_u)

    alpha = (3.0 / 16.0) * (-4.0 + 5.0 * fa_u * fa_u)

    Pplus = (
        0.25 * (M_L + 1.0)**2
        * (
            (2.0 - M_L)
            + 16.0 * alpha * M_L * 0.25 * (M_L - 1.0)**2
        )
    )

    Pminus = (
        -0.25 * (M_R - 1.0)**2
        * (
            (-2.0 - M_R)
            + 16.0 * alpha * M_R * 0.25 * (M_R + 1.0)**2
        )
    )

    Phalf = (
        Pplus * p_L
        + Pminus * p_R
        - K_u * Pplus * Pminus
        * (rho_L + rho_R)
        * a_half
        * fa_u
        * (vR - vL)
    )

    # ------------------------------------------------------------
    # Upwind velocity and theta
    # ------------------------------------------------------------
    selector = mdothalf > 0.0

    if normal == "z":
        u_up = numpy.where(selector, uL, uR)
        w_up = numpy.where(selector, vL, vR)
    else:
        w_up = numpy.where(selector, wL, wR)
        u_up = numpy.where(selector, vL, vR)

    theta_L = U_L[idx_theta] / rho_L
    theta_R = U_R[idx_theta] / rho_R
    theta_up = numpy.where(selector, theta_L, theta_R)

    # ------------------------------------------------------------
    # Build AUSM+up flux
    # ------------------------------------------------------------
    F = numpy.zeros_like(U_L, dtype=dtype)

    F[idx_rho] = mdothalf
    F[idx_u] = mdothalf * u_up
    F[idx_w] = mdothalf * w_up
    F[mom_n] += Phalf
    F[idx_theta] = mdothalf * theta_up

    return F
 
def rusanov_flux(
    U_L, U_R,                 # (neq, N) left/right conserved states
    p_L, p_R,                 # (N,) pressures on L/R
    gamma,                    # heat_capacity_ratio
    *,
    idx_rho, idx_u, idx_w, idx_theta,
    normal="z", p_base_L=None, p_base_R=None):
    """
    Vectorized Rusanov (local Lax-Friedrichs) interfacial flux.

    Returns
    -------
    F : ndarray, shape (neq, N)
        Numerical flux at the interface.
    """
    neq, N = U_L.shape
    dtype = U_L.dtype

    rho_L = U_L[idx_rho]
    rho_R = U_R[idx_rho]

    # Primitive velocities
    u_L = U_L[idx_u] / rho_L
    u_R = U_R[idx_u] / rho_R
    w_L = U_L[idx_w] / rho_L
    w_R = U_R[idx_w] / rho_R

    # Select normal velocity and normal-momentum index
    if normal == "z":
        vL = w_L
        vR = w_R
        mom_n = idx_w
    elif normal == "x":
        vL = u_L
        vR = u_R
        mom_n = idx_u
    else:
        raise ValueError("normal must be 'z' or 'x'")

    # Sound speed
    a_L = numpy.sqrt(gamma * p_L / rho_L)
    a_R = numpy.sqrt(gamma * p_R / rho_R)

    # Physical flux from left state
    F_L = numpy.zeros_like(U_L, dtype=dtype)
    F_L[idx_rho]   = rho_L * vL
    F_L[idx_u]     = rho_L * u_L * vL
    F_L[idx_w]     = rho_L * w_L * vL
    F_L[idx_theta] = U_L[idx_theta] * vL   # = (rho*theta) * v_n
    F_L[mom_n]    += p_L

    # Physical flux from right state
    F_R = numpy.zeros_like(U_R, dtype=dtype)
    F_R[idx_rho]   = rho_R * vR
    F_R[idx_u]     = rho_R * u_R * vR
    F_R[idx_w]     = rho_R * w_R * vR
    F_R[idx_theta] = U_R[idx_theta] * vR
    F_R[mom_n]    += p_R

    # Rusanov dissipation speed
    smax = numpy.maximum(numpy.abs(vL) + a_L, numpy.abs(vR) + a_R)

    # Numerical flux
    F = 0.5 * (F_L + F_R) - 0.5 * smax * (U_R - U_L)

    return F.astype(dtype, copy=False)



def rhs_bubble(Q, geom, mtrx, nbsolpts, nb_elements_x, nb_elements_z):

   # ============================================================
   # Switches
   # ============================================================
   USE_WELL_BALANCED = True
   # True:
   #    Q entering this function is deviation state:
   #       Q = Q_total - Q_base
   #
   # False:
   #    Q entering this function is physical state:
   #       Q = Q_total

   FLUX_NAME = "rusanov"  # "ausm_plus_up" or "rusanov"
   # Options:
   #    "rusanov"
   #    "ausm_plus_up"

   theta0 = 303.15

   # ============================================================
   # Hydrostatic base helper
   # ============================================================
   def hydrostatic_base_at_z(z):
      exner = 1.0 - gravity * z / (cpd * theta0)

      if numpy.any(exner <= 0.0):
         raise ValueError("Non-positive Exner pressure in hydrostatic base state.")

      p_base = p0 * exner**(cpd / Rd)
      rho_base = p0 / (Rd * theta0) * exner**(cvd / Rd)

      return rho_base, p_base

   # ============================================================
   # Build hydrostatic reference state at solution points
   # ============================================================
   Q_base = numpy.zeros_like(Q)

   rho_base, base_pressure = hydrostatic_base_at_z(geom.X3)

   Q_base[idx_2d_rho, :, :]       = rho_base
   Q_base[idx_2d_rho_u, :, :]     = 0.0
   Q_base[idx_2d_rho_w, :, :]     = 0.0
   Q_base[idx_2d_rho_theta, :, :] = rho_base * theta0

   # ============================================================
   # Interpret input Q
   # ============================================================
   if USE_WELL_BALANCED:
      # WB run:
      # Q is deviation state.
      Q_dev = Q
      Q_total = Q_dev + Q_base
   else:
      # Non-WB run:
      # Q is physical total state.
      Q_total = Q

   # ============================================================
   # Numerical flux selector
   # ============================================================
   def compute_num_flux(UL, UR, pL, pR, normal, bp_L=None, bp_R=None):

      if FLUX_NAME == "rusanov":
         return rusanov_flux(
            UL,
            UR,
            pL,
            pR,
            gamma=heat_capacity_ratio,
            idx_rho=idx_2d_rho,
            idx_u=idx_2d_rho_u,
            idx_w=idx_2d_rho_w,
            idx_theta=idx_2d_rho_theta,
            normal=normal,
            p_base_L=bp_L,
            p_base_R=bp_R,
         )

      elif FLUX_NAME == "ausm_plus_up":
         return ausm_plus_up_flux(
            UL,
            UR,
            pL,
            pR,
            gamma=heat_capacity_ratio,
            idx_rho=idx_2d_rho,
            idx_u=idx_2d_rho_u,
            idx_w=idx_2d_rho_w,
            idx_theta=idx_2d_rho_theta,
            normal=normal,
            p_base_L=bp_L,
            p_base_R=bp_R,
         )

      else:
         raise ValueError("Unknown FLUX_NAME. Use 'rusanov' or 'ausm_plus_up'.")

   # ============================================================
   # Physical RHS
   # ============================================================
   def rhs_physical(Q_phys, use_wb_interface, pass_base_pressure_to_flux):

      datatype = Q_phys.dtype
      nb_equations = Q_phys.shape[0]

      nb_interfaces_x = nb_elements_x + 1
      nb_interfaces_z = nb_elements_z + 1

      flux_x1 = numpy.zeros_like(Q_phys, dtype=datatype)
      flux_x3 = numpy.zeros_like(Q_phys, dtype=datatype)

      df1_dx1 = numpy.zeros_like(Q_phys, dtype=datatype)
      df3_dx3 = numpy.zeros_like(Q_phys, dtype=datatype)

      kfaces_flux = numpy.zeros(
         (nb_equations, nb_elements_z, 2, nbsolpts * nb_elements_x),
         dtype=datatype,
      )

      kfaces_var = numpy.zeros_like(kfaces_flux)
      kfaces_dev = numpy.zeros_like(kfaces_flux)
      kfaces_base = numpy.zeros_like(kfaces_flux)

      kfaces_bp = numpy.zeros(
         (nb_elements_z, 2, nbsolpts * nb_elements_x),
         dtype=datatype,
      )

      ifaces_flux = numpy.zeros(
         (nb_equations, nb_elements_x, nbsolpts * nb_elements_z, 2),
         dtype=datatype,
      )

      ifaces_var = numpy.zeros_like(ifaces_flux)
      ifaces_dev = numpy.zeros_like(ifaces_flux)
      ifaces_base = numpy.zeros_like(ifaces_flux)

      ifaces_bp = numpy.zeros(
         (nb_elements_x, nbsolpts * nb_elements_z, 2),
         dtype=datatype,
      )

      # ---------------------------------------------------------
      # Physical primitive variables at solution points
      # ---------------------------------------------------------
      rho = Q_phys[idx_2d_rho, :, :]

      if numpy.min(rho) <= 0.0:
         raise ValueError("Non-positive physical density in rhs_physical.")

      rho_theta = Q_phys[idx_2d_rho_theta, :, :]

      if numpy.min(rho_theta) <= 0.0:
         raise ValueError("Non-positive physical rho*theta in rhs_physical.")

      uu = Q_phys[idx_2d_rho_u, :, :] / rho
      ww = Q_phys[idx_2d_rho_w, :, :] / rho

      pressure = p0 * (
         (Rd / p0) * rho_theta
      )**(cpd / cvd)

      # ---------------------------------------------------------
      # Physical fluxes at solution points
      # ---------------------------------------------------------
      flux_x1[idx_2d_rho, :, :]       = Q_phys[idx_2d_rho_u, :, :]
      flux_x1[idx_2d_rho_u, :, :]     = Q_phys[idx_2d_rho_u, :, :] * uu + pressure
      flux_x1[idx_2d_rho_w, :, :]     = Q_phys[idx_2d_rho_u, :, :] * ww
      flux_x1[idx_2d_rho_theta, :, :] = Q_phys[idx_2d_rho_theta, :, :] * uu

      flux_x3[idx_2d_rho, :, :]       = Q_phys[idx_2d_rho_w, :, :]
      flux_x3[idx_2d_rho_u, :, :]     = Q_phys[idx_2d_rho_w, :, :] * uu
      flux_x3[idx_2d_rho_w, :, :]     = Q_phys[idx_2d_rho_w, :, :] * ww + pressure
      flux_x3[idx_2d_rho_theta, :, :] = Q_phys[idx_2d_rho_theta, :, :] * ww

      # ---------------------------------------------------------
      # Local deviation relative to hydrostatic base
      # ---------------------------------------------------------
      Q_local_dev = Q_phys - Q_base

      standard_slice = numpy.arange(nbsolpts)

      # =========================================================
      # z-direction interface reconstruction
      # =========================================================
      for elem in range(nb_elements_z):
         epais = elem * nbsolpts + standard_slice

         if use_wb_interface:

            # ---------------------------------------------------
            # Reconstruct only deviation
            # ---------------------------------------------------
            kfaces_dev[:, elem, 0, :] = (
               mtrx.extrap_down @ Q_local_dev[:, epais, :]
            )

            kfaces_dev[:, elem, 1, :] = (
               mtrx.extrap_up @ Q_local_dev[:, epais, :]
            )

            # ---------------------------------------------------
            # Analytic hydrostatic base at vertical faces
            # ---------------------------------------------------
            z_down = geom.itf_Z[elem]
            z_up   = geom.itf_Z[elem + 1]

            rho_down, p_down = hydrostatic_base_at_z(z_down)
            rho_up,   p_up   = hydrostatic_base_at_z(z_up)

            kfaces_base[idx_2d_rho, elem, 0, :]       = rho_down
            kfaces_base[idx_2d_rho_u, elem, 0, :]     = 0.0
            kfaces_base[idx_2d_rho_w, elem, 0, :]     = 0.0
            kfaces_base[idx_2d_rho_theta, elem, 0, :] = rho_down * theta0

            kfaces_base[idx_2d_rho, elem, 1, :]       = rho_up
            kfaces_base[idx_2d_rho_u, elem, 1, :]     = 0.0
            kfaces_base[idx_2d_rho_w, elem, 1, :]     = 0.0
            kfaces_base[idx_2d_rho_theta, elem, 1, :] = rho_up * theta0

            # ---------------------------------------------------
            # Physical interface state:
            #    U_face = U_base_analytic_face + reconstruct(U_dev)
            # ---------------------------------------------------
            kfaces_var[:, elem, 0, :] = (
               kfaces_base[:, elem, 0, :] + kfaces_dev[:, elem, 0, :]
            )

            kfaces_var[:, elem, 1, :] = (
               kfaces_base[:, elem, 1, :] + kfaces_dev[:, elem, 1, :]
            )

            # Base pressure at faces
            kfaces_bp[elem, 0, :] = p_down
            kfaces_bp[elem, 1, :] = p_up

         else:

            # ---------------------------------------------------
            # Non-WB: reconstruct physical state directly
            # ---------------------------------------------------
            kfaces_var[:, elem, 0, :] = (
               mtrx.extrap_down @ Q_phys[:, epais, :]
            )

            kfaces_var[:, elem, 1, :] = (
               mtrx.extrap_up @ Q_phys[:, epais, :]
            )

            kfaces_bp[elem, 0, :] = 0.0
            kfaces_bp[elem, 1, :] = 0.0

      # =========================================================
      # x-direction interface reconstruction
      # =========================================================
      for elem in range(nb_elements_x):
         epais = elem * nbsolpts + standard_slice

         if use_wb_interface:

            # ---------------------------------------------------
            # Reconstruct only deviation
            # ---------------------------------------------------
            ifaces_dev[:, elem, :, 0] = (
               Q_local_dev[:, :, epais] @ mtrx.extrap_west
            )

            ifaces_dev[:, elem, :, 1] = (
               Q_local_dev[:, :, epais] @ mtrx.extrap_east
            )

            # ---------------------------------------------------
            # Analytic hydrostatic base at x-faces
            #
            # Base depends only on z. We compute z at the x-face
            # and evaluate the hydrostatic state analytically.
            # ---------------------------------------------------
            z_west = geom.X3[:, epais] @ mtrx.extrap_west
            z_east = geom.X3[:, epais] @ mtrx.extrap_east

            rho_west, p_west = hydrostatic_base_at_z(z_west)
            rho_east, p_east = hydrostatic_base_at_z(z_east)

            ifaces_base[idx_2d_rho, elem, :, 0]       = rho_west
            ifaces_base[idx_2d_rho_u, elem, :, 0]     = 0.0
            ifaces_base[idx_2d_rho_w, elem, :, 0]     = 0.0
            ifaces_base[idx_2d_rho_theta, elem, :, 0] = rho_west * theta0

            ifaces_base[idx_2d_rho, elem, :, 1]       = rho_east
            ifaces_base[idx_2d_rho_u, elem, :, 1]     = 0.0
            ifaces_base[idx_2d_rho_w, elem, :, 1]     = 0.0
            ifaces_base[idx_2d_rho_theta, elem, :, 1] = rho_east * theta0

            # ---------------------------------------------------
            # Physical interface state:
            #    U_face = U_base_analytic_face + reconstruct(U_dev)
            # ---------------------------------------------------
            ifaces_var[:, elem, :, 0] = (
               ifaces_base[:, elem, :, 0] + ifaces_dev[:, elem, :, 0]
            )

            ifaces_var[:, elem, :, 1] = (
               ifaces_base[:, elem, :, 1] + ifaces_dev[:, elem, :, 1]
            )

            ifaces_bp[elem, :, 0] = p_west
            ifaces_bp[elem, :, 1] = p_east

         else:

            # ---------------------------------------------------
            # Non-WB: reconstruct physical state directly
            # ---------------------------------------------------
            ifaces_var[:, elem, :, 0] = (
               Q_phys[:, :, epais] @ mtrx.extrap_west
            )

            ifaces_var[:, elem, :, 1] = (
               Q_phys[:, :, epais] @ mtrx.extrap_east
            )

            ifaces_bp[elem, :, 0] = 0.0
            ifaces_bp[elem, :, 1] = 0.0

      # ---------------------------------------------------------
      # Interface pressure from physical interface states
      # ---------------------------------------------------------
      ifaces_rho_theta = ifaces_var[idx_2d_rho_theta]
      kfaces_rho_theta = kfaces_var[idx_2d_rho_theta]

      if numpy.min(ifaces_rho_theta) <= 0.0:
         raise ValueError("Non-positive rho*theta at x-interface.")

      if numpy.min(kfaces_rho_theta) <= 0.0:
         raise ValueError("Non-positive rho*theta at z-interface.")

      ifaces_pres = p0 * (
         ifaces_rho_theta * Rd / p0
      )**(cpd / cvd)

      kfaces_pres = p0 * (
         kfaces_rho_theta * Rd / p0
      )**(cpd / cvd)

      # =========================================================
      # Boundary treatment
      # =========================================================

      # zeros flux BCs everywhere first
      kfaces_flux[:, 0, 0, :]  = 0.0
      kfaces_flux[:, -1, 1, :] = 0.0

      if not geom.xperiodic:
         ifaces_flux[:, 0, :, 0]  = 0.0
         ifaces_flux[:, -1, :, 1] = 0.0

      # pressure contribution at vertical boundaries
      kfaces_flux[idx_2d_rho_w, 0, 0, :]  = kfaces_pres[0, 0, :]
      kfaces_flux[idx_2d_rho_w, -1, 1, :] = kfaces_pres[-1, 1, :]

      # pressure contribution at horizontal boundaries
      ifaces_flux[idx_2d_rho_u, 0, :, 0]  = ifaces_pres[0, :, 0]
      ifaces_flux[idx_2d_rho_u, -1, :, 1] = ifaces_pres[-1, :, 1]

      # =========================================================
      # z-direction numerical fluxes
      # =========================================================
      for itf in range(1, nb_interfaces_z - 1):
         left = itf - 1
         right = itf

         UL = kfaces_var[:, left, 1, :]
         UR = kfaces_var[:, right, 0, :]

         pL = kfaces_pres[left, 1, :]
         pR = kfaces_pres[right, 0, :]

         if pass_base_pressure_to_flux:
            bp_L = kfaces_bp[left, 1, :]
            bp_R = kfaces_bp[right, 0, :]
         else:
            bp_L = None
            bp_R = None

         flux = compute_num_flux(
            UL,
            UR,
            pL,
            pR,
            normal="z",
            bp_L=bp_L,
            bp_R=bp_R,
         )

         kfaces_flux[:, right, 0, :] = flux
         kfaces_flux[:, left, 1, :]  = flux

      # =========================================================
      # x-direction numerical fluxes
      # =========================================================
      start = 0 if geom.xperiodic else 1

      for itf in range(start, nb_interfaces_x - 1):
         left = itf - 1
         right = itf

         UL = ifaces_var[:, left, :, 1]
         UR = ifaces_var[:, right, :, 0]

         pL = ifaces_pres[left, :, 1]
         pR = ifaces_pres[right, :, 0]

         if pass_base_pressure_to_flux:
            bp_L = ifaces_bp[left, :, 1]
            bp_R = ifaces_bp[right, :, 0]
         else:
            bp_L = None
            bp_R = None

         flux = compute_num_flux(
            UL,
            UR,
            pL,
            pR,
            normal="x",
            bp_L=bp_L,
            bp_R=bp_R,
         )

         ifaces_flux[:, right, :, 0] = flux
         ifaces_flux[:, left, :, 1]  = flux

      if geom.xperiodic:
         ifaces_flux[:, 0, :, 0] = ifaces_flux[:, -1, :, 1]

      # =========================================================
      # Compute derivatives
      # =========================================================
      for elem in range(nb_elements_z):
         epais = elem * nbsolpts + standard_slice

         factor = 2.0 / geom.Δx3
         if elem < geom.nb_elements_relief_layer:
            factor = 2.0 / geom.relief_layer_delta

         df3_dx3[:, epais, :] = (
            mtrx.diff_solpt @ flux_x3[:, epais, :]
            + mtrx.correction @ kfaces_flux[:, elem, :, :]
         ) * factor

      for elem in range(nb_elements_x):
         epais = elem * nbsolpts + standard_slice

         df1_dx1[:, :, epais] = (
            flux_x1[:, :, epais] @ mtrx.diff_solpt.T
            + ifaces_flux[:, elem, :, :] @ mtrx.correction.T
         ) * 2.0 / geom.Δx1

      # =========================================================
      # Assemble RHS
      # =========================================================
      rhs_out = -(df1_dx1 + df3_dx3)

      # Gravity source uses physical density
      rhs_out[idx_2d_rho_w, :, :] -= Q_phys[idx_2d_rho, :, :] * gravity

      return rhs_out

   # ============================================================
   # Return selected RHS
   # ============================================================
   if USE_WELL_BALANCED:

      t_rhs = rhs_physical(
         Q_total,
         use_wb_interface=True,
         pass_base_pressure_to_flux=True,
      )

      b_rhs = rhs_physical(
         Q_base,
         use_wb_interface=True,
         pass_base_pressure_to_flux=True,
      )

      return t_rhs - b_rhs

   else:

      return rhs_physical(
         Q_total,
         use_wb_interface=False,
         pass_base_pressure_to_flux=False,
      )
      

      
