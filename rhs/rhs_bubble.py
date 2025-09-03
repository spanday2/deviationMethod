import numpy

from common.definitions import idx_2d_rho, idx_2d_rho_u, idx_2d_rho_w, idx_2d_rho_theta,  \
                               p0, Rd, cpd, cvd, heat_capacity_ratio, gravity


def ausm_plus_flux(
    U_L, U_R,                 # (neq, N) left/right conserved states at the interface
    p_L, p_R,                 # (N,) pressures on L/R
    gamma,                    # heat_capacity_ratio
    *,
    idx_rho, idx_u, idx_w, idx_E,
    normal="z",               # "z" for vertical faces (normal=w), "x" for horizontal (normal=u)
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


def rhs_bubble(Q, geom, mtrx, nbsolpts, nb_elements_x, nb_elements_z):
   
   def rhs(Q, geom, mtrx, nbsolpts, nb_elements_x, nb_elements_z):
      datatype = Q.dtype
      nb_equations = Q.shape[0] # Number of constituent Euler equations.  Probably 6.

      nb_interfaces_x = nb_elements_x + 1
      nb_interfaces_z = nb_elements_z + 1

      flux_x1 = numpy.empty_like(Q, dtype=datatype)
      flux_x3 = numpy.empty_like(Q, dtype=datatype)

      df1_dx1 = numpy.empty_like(Q, dtype=datatype)
      df3_dx3 = numpy.empty_like(Q, dtype=datatype)

      kfaces_flux = numpy.empty((nb_equations, nb_elements_z, 2, nbsolpts*nb_elements_x), dtype=datatype)
      kfaces_var  = numpy.empty((nb_equations, nb_elements_z, 2, nbsolpts*nb_elements_x), dtype=datatype)

      ifaces_flux = numpy.empty((nb_equations, nb_elements_x, nbsolpts*nb_elements_z, 2), dtype=datatype)
      ifaces_var  = numpy.empty((nb_equations, nb_elements_x, nbsolpts*nb_elements_z, 2), dtype=datatype)

      # --- Unpack physical variables
      rho      = Q[idx_2d_rho,:,:]
      uu       = Q[idx_2d_rho_u,:,:] / rho
      ww       = Q[idx_2d_rho_w,:,:] / rho
      pressure = p0 * numpy.exp((cpd/cvd) * numpy.log((Rd/p0)*Q[idx_2d_rho_theta, :, :]))

      # --- Compute the fluxes
      flux_x1[idx_2d_rho,:,:]       = Q[idx_2d_rho_u,:,:]
      flux_x1[idx_2d_rho_u,:,:]     = Q[idx_2d_rho_u,:,:] * uu + pressure
      flux_x1[idx_2d_rho_w,:,:]     = Q[idx_2d_rho_u,:,:] * ww
      flux_x1[idx_2d_rho_theta,:,:] = Q[idx_2d_rho_theta,:,:] * uu

      flux_x3[idx_2d_rho,:,:]       = Q[idx_2d_rho_w,:,:]
      flux_x3[idx_2d_rho_u,:,:]     = Q[idx_2d_rho_w,:,:] * uu
      flux_x3[idx_2d_rho_w,:,:]     = Q[idx_2d_rho_w,:,:] * ww + pressure
      flux_x3[idx_2d_rho_theta,:,:] = Q[idx_2d_rho_theta,:,:] * ww


   #   print(numpy.max(flux_x3[idx_2d_rho_w,:,:])); exit(0)

      # --- Interpolate to the element interface
      standard_slice = numpy.arange(nbsolpts)
      for elem in range(nb_elements_z):
         epais = elem * nbsolpts + standard_slice

         kfaces_var[:,elem,0,:] = mtrx.extrap_down @ Q[:,epais,:]
         kfaces_var[:,elem,1,:] = mtrx.extrap_up @ Q[:,epais,:]

      for elem in range(nb_elements_x):
         epais = elem * nbsolpts + standard_slice

         ifaces_var[:,elem,:,0] = Q[:,:,epais] @ mtrx.extrap_west
         ifaces_var[:,elem,:,1] = Q[:,:,epais] @ mtrx.extrap_east

      # --- Interface pressure
      ifaces_pres = p0 * (ifaces_var[idx_2d_rho_theta] * Rd / p0)**(cpd / cvd)
      kfaces_pres = p0 * (kfaces_var[idx_2d_rho_theta] * Rd / p0)**(cpd / cvd)

      # --- Bondary treatement

      # zeros flux BCs everywhere ...
      kfaces_flux[:,0,0,:]  = 0.0
      kfaces_flux[:,-1,1,:] = 0.0

      # Skip periodic faces
      if not geom.xperiodic:
         ifaces_flux[:, 0,:,0] = 0.0
         ifaces_flux[:,-1,:,1] = 0.0

      # except for momentum eqs where pressure is extrapolated to BCs.
      kfaces_flux[idx_2d_rho_w, 0, 0, :] = kfaces_pres[ 0, 0, :]
      kfaces_flux[idx_2d_rho_w,-1, 1, :] = kfaces_pres[-1, 1, :]

      ifaces_flux[idx_2d_rho_u, 0,:,0] = ifaces_pres[0,:,0]  # TODO : pour les cas théoriques seulement ...
      ifaces_flux[idx_2d_rho_u,-1,:,1] = ifaces_pres[-1,:,1]

      # --- Common AUSM fluxes
      for itf in range(1, nb_interfaces_z - 1):

         left  = itf - 1
         right = itf

         # Gather left/right slices (shape (neq, N))
         UL = kfaces_var[:, left,  1, :]
         UR = kfaces_var[:, right, 0, :]

         pL = kfaces_pres[left,  1, :]
         pR = kfaces_pres[right, 0, :]

         flux = ausm_plus_flux(
            UL, UR, pL, pR,
            gamma=heat_capacity_ratio,
            idx_rho=idx_2d_rho, idx_u=idx_2d_rho_u, idx_w=idx_2d_rho_w, idx_E=idx_2d_rho_theta,
            normal="z"
         )

         kfaces_flux[:, right, 0, :] = flux
         kfaces_flux[:, left,  1, :] = flux  # mirror


      start = 0 if geom.xperiodic else 1
      for itf in range(start, nb_interfaces_x - 1):

         left  = itf - 1
         right = itf

         UL = ifaces_var[:, left,  :, 1]   # (neq, N)
         UR = ifaces_var[:, right, :, 0]

         pL = ifaces_pres[left,  :, 1]     # (N,)
         pR = ifaces_pres[right, :, 0]

         flux = ausm_plus_flux(
            UL, UR, pL, pR,
            gamma=heat_capacity_ratio,
            idx_rho=idx_2d_rho, idx_u=idx_2d_rho_u, idx_w=idx_2d_rho_w, idx_E=idx_2d_rho_theta,
            normal="x"
         )

         ifaces_flux[:, right, :, 0] = flux
         ifaces_flux[:, left,  :, 1] = flux  # mirror

      if geom.xperiodic:
         ifaces_flux[:, 0, :, 0] = ifaces_flux[:, -1, :, 1]

      # --- Compute the derivatives
      for elem in range(nb_elements_z):
         epais = elem * nbsolpts + standard_slice
         factor = 2.0 / geom.Δx3
         if elem < geom.nb_elements_relief_layer:
            factor = 2.0 / geom.relief_layer_delta

         df3_dx3[:, epais, :] = \
            (mtrx.diff_solpt @ flux_x3[:, epais, :] + mtrx.correction @ kfaces_flux[:, elem, :, :]) * factor

      for elem in range(nb_elements_x):
         epais = elem * nbsolpts + numpy.arange(nbsolpts)

         df1_dx1[:,:,epais] = (flux_x1[:,:,epais] @ mtrx.diff_solpt.T + ifaces_flux[:,elem,:,:] @ mtrx.correction.T) * \
                              2.0/geom.Δx1

      # --- Assemble the right-hand sides
      rhs = - ( df1_dx1 + df3_dx3 )
     

      rhs[idx_2d_rho_w,:,:] -= Q[idx_2d_rho,:,:] * gravity


      return rhs
   
   t_rhs = rhs(Q, geom, mtrx, nbsolpts, nb_elements_x, nb_elements_z)

   
   
   return t_rhs

      
