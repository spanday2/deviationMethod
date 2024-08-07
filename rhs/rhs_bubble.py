import numpy

from common.definitions import idx_2d_rho, idx_2d_rho_u, idx_2d_rho_w, idx_2d_rho_theta,  \
                               p0, Rd, cpd, cvd, heat_capacity_ratio, gravity

def rhs_bubble(Q, geom, mtrx, nbsolpts, nb_elements_x, nb_elements_z):

   # Q = deltaQ + Q_tilda

   datatype = Q.dtype
   nb_equations = Q.shape[0] # Number of constituent Euler equations.  Probably 6.

   nb_interfaces_x = nb_elements_x + 1
   nb_interfaces_z = nb_elements_z + 1

   flux_x1, t_flux_x1 = [numpy.empty_like(Q, dtype=datatype) for _ in range(2)]
   flux_x3, t_flux_x3 = [numpy.empty_like(Q, dtype=datatype) for _ in range(2)]

   df1_dx1, t_df1_dx1 = [numpy.empty_like(Q, dtype=datatype) for _ in range(2)]
   df3_dx3, t_df3_dx3 = [numpy.empty_like(Q, dtype=datatype) for _ in range(2)]

   kfaces_flux, t_kfaces_flux = [numpy.empty((nb_equations, nb_elements_z, 2, nbsolpts*nb_elements_x), dtype=datatype) for _ in range(2)]
   kfaces_var, t_kfaces_var  = [numpy.empty((nb_equations, nb_elements_z, 2, nbsolpts*nb_elements_x), dtype=datatype) for _ in range(2)]

   ifaces_flux, t_ifaces_flux = [numpy.empty((nb_equations, nb_elements_x, nbsolpts*nb_elements_z, 2), dtype=datatype) for _ in range(2)]
   ifaces_var, t_ifaces_var  = [numpy.empty((nb_equations, nb_elements_x, nbsolpts*nb_elements_z, 2), dtype=datatype) for _ in range(2)]

   # --- Unpack physical variables
   rho      = Q[idx_2d_rho,:,:]
   uu       = Q[idx_2d_rho_u,:,:] / rho
   ww       = Q[idx_2d_rho_w,:,:] / rho
   pressure = p0 * numpy.exp((cpd/cvd) * numpy.log((Rd/p0)*Q[idx_2d_rho_theta, :, :]))

   # t_rho      = Q_tilda[idx_2d_rho,:,:]
   # t_uu       = Q_tilda[idx_2d_rho_u,:,:] / t_rho
   # t_ww       = Q_tilda[idx_2d_rho_w,:,:] / t_rho
   # t_pressure = p0 * numpy.exp((cpd/cvd) * numpy.log((Rd/p0)*Q_tilda[idx_2d_rho_theta, :, :]))

   # --- Compute the fluxes
   flux_x1[idx_2d_rho,:,:]       = Q[idx_2d_rho_u,:,:]
   flux_x1[idx_2d_rho_u,:,:]     = Q[idx_2d_rho_u,:,:] * uu + pressure
   flux_x1[idx_2d_rho_w,:,:]     = Q[idx_2d_rho_u,:,:] * ww
   flux_x1[idx_2d_rho_theta,:,:] = Q[idx_2d_rho_theta,:,:] * uu

   flux_x3[idx_2d_rho,:,:]       = Q[idx_2d_rho_w,:,:]
   flux_x3[idx_2d_rho_u,:,:]     = Q[idx_2d_rho_w,:,:] * uu
   flux_x3[idx_2d_rho_w,:,:]     = Q[idx_2d_rho_w,:,:] * ww + pressure
   flux_x3[idx_2d_rho_theta,:,:] = Q[idx_2d_rho_theta,:,:] * ww

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

      # Left state
      a_L = numpy.sqrt(heat_capacity_ratio * kfaces_pres[left, 1, :] / kfaces_var[idx_2d_rho, left, 1, :])
      M_L = kfaces_var[idx_2d_rho_w, left, 1, :] / (kfaces_var[idx_2d_rho, left, 1, :] * a_L)

      # Right state
      a_R = numpy.sqrt(heat_capacity_ratio * kfaces_pres[right, 0, :] / kfaces_var[idx_2d_rho, right, 0, :])
      M_R = kfaces_var[idx_2d_rho_w, right, 0, :] / (kfaces_var[idx_2d_rho, right, 0, :] * a_R)

      M = 0.25 * (( M_L + 1.)**2 - (M_R - 1.)**2)

      kfaces_flux[:,right,0,:] = (kfaces_var[:,left,1,:] * numpy.maximum(0., M) * a_L) + \
                                 (kfaces_var[:,right,0,:] * numpy.minimum(0., M) * a_R)
      kfaces_flux[idx_2d_rho_w,right,0,:] += 0.5 * ((1. + M_L) * kfaces_pres[left,1,:] + \
                                                    (1. - M_R) * kfaces_pres[right,0,:])

      kfaces_flux[:,left,1,:] = kfaces_flux[:,right,0,:]


   start = 0 if geom.xperiodic else 1
   for itf in range(start, nb_interfaces_x - 1):

      left  = itf - 1
      right = itf

      # Left state
      a_L = numpy.sqrt(heat_capacity_ratio * ifaces_pres[left, :, 1] / ifaces_var[idx_2d_rho, left, :, 1])
      M_L = ifaces_var[idx_2d_rho_u, left, :, 1] / (ifaces_var[idx_2d_rho, left, :, 1] * a_L)

      # Right state
      a_R = numpy.sqrt(heat_capacity_ratio * ifaces_pres[right, :, 0] / ifaces_var[idx_2d_rho, right, :, 0])
      M_R = ifaces_var[idx_2d_rho_u, right, :, 0] / ( ifaces_var[idx_2d_rho, right, :, 0] * a_R)

      M = 0.25 * ((M_L + 1.)**2 - (M_R - 1.)**2)

      ifaces_flux[:,right,:,0] = (ifaces_var[:,left,:,1] * numpy.maximum(0., M) * a_L) + \
                                 (ifaces_var[:,right,:,0] * numpy.minimum(0., M) * a_R)
      ifaces_flux[idx_2d_rho_u,right,:,0] += 0.5 * ((1. + M_L) * ifaces_pres[left,:,1] + \
                                                    (1. - M_R) * ifaces_pres[right,:,0])

      ifaces_flux[:,left,:,1] = ifaces_flux[:,right,:,0]

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

   # TODO : Add sources terms for Brikman penalization
   # It may be better to do this elementwise...
   if geom.nb_elements_relief_layer > 1:

      end = geom.nb_elements_relief_layer * nbsolpts
      etac = 1.0 # 1e-1

      normal_flux = numpy.where( \
            geom.relief_boundary_mask,
            geom.normals_x * df1_dx1[idx_2d_rho_u, :end, :] + geom.normals_z * df3_dx3[idx_2d_rho_w, :end, :],
            0.0)

      rhs[idx_2d_rho_u, :end, :] = numpy.where( \
            geom.relief_mask, -(1.0 / etac) * normal_flux * geom.normals_x, rhs[idx_2d_rho_u, :end, :])
      rhs[idx_2d_rho_w, :end, :] = numpy.where( \
            geom.relief_mask, -(1.0 / etac) * normal_flux * geom.normals_z, rhs[idx_2d_rho_w, :end, :])


   #--------------------------------------------------------------------------------------------------------------------------------------------#
   #--------------------------------------------------------- Q_tilda --------------------------------------------------------------------------#
   #--------------------------------------------------------------------------------------------------------------------------------------------#
    
   # # --- Compute the fluxes
   # t_flux_x1[idx_2d_rho,:,:]       = Q_tilda[idx_2d_rho_u,:,:]
   # t_flux_x1[idx_2d_rho_u,:,:]     = Q_tilda[idx_2d_rho_u,:,:] * t_uu + t_pressure
   # t_flux_x1[idx_2d_rho_w,:,:]     = Q_tilda[idx_2d_rho_u,:,:] * t_ww
   # t_flux_x1[idx_2d_rho_theta,:,:] = Q_tilda[idx_2d_rho_theta,:,:] * t_uu

   # t_flux_x3[idx_2d_rho,:,:]       = Q_tilda[idx_2d_rho_w,:,:]
   # t_flux_x3[idx_2d_rho_u,:,:]     = Q_tilda[idx_2d_rho_w,:,:] * t_uu
   # t_flux_x3[idx_2d_rho_w,:,:]     = Q_tilda[idx_2d_rho_w,:,:] * t_ww + t_pressure
   # t_flux_x3[idx_2d_rho_theta,:,:] = Q_tilda[idx_2d_rho_theta,:,:] * t_ww

   # # --- Interpolate to the element interface
   # standard_slice = numpy.arange(nbsolpts)
   # for elem in range(nb_elements_z):
   #    epais = elem * nbsolpts + standard_slice

   #    t_kfaces_var[:,elem,0,:] = mtrx.extrap_down @ Q_tilda[:,epais,:]
   #    t_kfaces_var[:,elem,1,:] = mtrx.extrap_up @ Q_tilda[:,epais,:]

   # for elem in range(nb_elements_x):
   #    epais = elem * nbsolpts + standard_slice

   #    t_ifaces_var[:,elem,:,0] = Q_tilda[:,:,epais] @ mtrx.extrap_west
   #    t_ifaces_var[:,elem,:,1] = Q_tilda[:,:,epais] @ mtrx.extrap_east

   # # --- Interface pressure
   # t_ifaces_pres = p0 * (t_ifaces_var[idx_2d_rho_theta] * Rd / p0)**(cpd / cvd)
   # t_kfaces_pres = p0 * (t_kfaces_var[idx_2d_rho_theta] * Rd / p0)**(cpd / cvd)

   # # --- Bondary treatement

   # # zeros flux BCs everywhere ...
   # t_kfaces_flux[:,0,0,:]  = 0.0
   # t_kfaces_flux[:,-1,1,:] = 0.0

   # # Skip periodic faces
   # if not geom.xperiodic:
   #    t_ifaces_flux[:, 0,:,0] = 0.0
   #    t_ifaces_flux[:,-1,:,1] = 0.0

   # # except for momentum eqs where pressure is extrapolated to BCs.
   # t_kfaces_flux[idx_2d_rho_w, 0, 0, :] = t_kfaces_pres[ 0, 0, :]
   # t_kfaces_flux[idx_2d_rho_w,-1, 1, :] = t_kfaces_pres[-1, 1, :]

   # t_ifaces_flux[idx_2d_rho_u, 0,:,0] = t_ifaces_pres[0,:,0]  # TODO : pour les cas théoriques seulement ...
   # t_ifaces_flux[idx_2d_rho_u,-1,:,1] = t_ifaces_pres[-1,:,1]

   # # --- Common AUSM fluxes
   # for itf in range(1, nb_interfaces_z - 1):

   #    left  = itf - 1
   #    right = itf

   #    # Left state
   #    a_L = numpy.sqrt(heat_capacity_ratio * t_kfaces_pres[left, 1, :] / t_kfaces_var[idx_2d_rho, left, 1, :])
   #    M_L = t_kfaces_var[idx_2d_rho_w, left, 1, :] / (t_kfaces_var[idx_2d_rho, left, 1, :] * a_L)

   #    # Right state
   #    a_R = numpy.sqrt(heat_capacity_ratio * t_kfaces_pres[right, 0, :] / t_kfaces_var[idx_2d_rho, right, 0, :])
   #    M_R = t_kfaces_var[idx_2d_rho_w, right, 0, :] / (t_kfaces_var[idx_2d_rho, right, 0, :] * a_R)

   #    M = 0.25 * (( M_L + 1.)**2 - (M_R - 1.)**2)

   #    t_kfaces_flux[:,right,0,:] = (t_kfaces_var[:,left,1,:] * numpy.maximum(0., M) * a_L) + \
   #                               (t_kfaces_var[:,right,0,:] * numpy.minimum(0., M) * a_R)
   #    t_kfaces_flux[idx_2d_rho_w,right,0,:] += 0.5 * ((1. + M_L) * t_kfaces_pres[left,1,:] + \
   #                                                  (1. - M_R) * t_kfaces_pres[right,0,:])

   #    t_kfaces_flux[:,left,1,:] = t_kfaces_flux[:,right,0,:]


   # start = 0 if geom.xperiodic else 1
   # for itf in range(start, nb_interfaces_x - 1):

   #    left  = itf - 1
   #    right = itf

   #    # Left state
   #    a_L = numpy.sqrt(heat_capacity_ratio * t_ifaces_pres[left, :, 1] / t_ifaces_var[idx_2d_rho, left, :, 1])
   #    M_L = t_ifaces_var[idx_2d_rho_u, left, :, 1] / (t_ifaces_var[idx_2d_rho, left, :, 1] * a_L)

   #    # Right state
   #    a_R = numpy.sqrt(heat_capacity_ratio * t_ifaces_pres[right, :, 0] / t_ifaces_var[idx_2d_rho, right, :, 0])
   #    M_R = t_ifaces_var[idx_2d_rho_u, right, :, 0] / ( t_ifaces_var[idx_2d_rho, right, :, 0] * a_R)

   #    M = 0.25 * ((M_L + 1.)**2 - (M_R - 1.)**2)

   #    t_ifaces_flux[:,right,:,0] = (t_ifaces_var[:,left,:,1] * numpy.maximum(0., M) * a_L) + \
   #                               (t_ifaces_var[:,right,:,0] * numpy.minimum(0., M) * a_R)
   #    t_ifaces_flux[idx_2d_rho_u,right,:,0] += 0.5 * ((1. + M_L) * t_ifaces_pres[left,:,1] + \
   #                                                  (1. - M_R) * t_ifaces_pres[right,:,0])

   #    t_ifaces_flux[:,left,:,1] = t_ifaces_flux[:,right,:,0]

   # if geom.xperiodic:
   #    t_ifaces_flux[:, 0, :, 0] = t_ifaces_flux[:, -1, :, 1]

   # # --- Compute the derivatives
   # for elem in range(nb_elements_z):
   #    epais = elem * nbsolpts + standard_slice
   #    factor = 2.0 / geom.Δx3
   #    if elem < geom.nb_elements_relief_layer:
   #       factor = 2.0 / geom.relief_layer_delta

   #    t_df3_dx3[:, epais, :] = \
   #       (mtrx.diff_solpt @ t_flux_x3[:, epais, :] + mtrx.correction @ t_kfaces_flux[:, elem, :, :]) * factor

   # for elem in range(nb_elements_x):
   #    epais = elem * nbsolpts + numpy.arange(nbsolpts)

   #    t_df1_dx1[:,:,epais] = (t_flux_x1[:,:,epais] @ mtrx.diff_solpt.T + t_ifaces_flux[:,elem,:,:] @ mtrx.correction.T) * \
   #                         2.0/geom.Δx1

   # # --- Assemble the right-hand sides
   # t_rhs = - ( t_df1_dx1 + t_df3_dx3 )

   # t_rhs[idx_2d_rho_w,:,:] -= Q_tilda[idx_2d_rho,:,:] * gravity

   # # TODO : Add sources terms for Brikman penalization
   # # It may be better to do this elementwise...
   # if geom.nb_elements_relief_layer > 1:

   #    end = geom.nb_elements_relief_layer * nbsolpts
   #    etac = 1.0 # 1e-1

   #    normal_flux = numpy.where( \
   #          geom.relief_boundary_mask,
   #          geom.normals_x * t_df1_dx1[idx_2d_rho_u, :end, :] + geom.normals_z * t_df3_dx3[idx_2d_rho_w, :end, :],
   #          0.0)

   #    t_rhs[idx_2d_rho_u, :end, :] = numpy.where( \
   #          geom.relief_mask, -(1.0 / etac) * normal_flux * geom.normals_x, t_rhs[idx_2d_rho_u, :end, :])
   #    t_rhs[idx_2d_rho_w, :end, :] = numpy.where( \
   #          geom.relief_mask, -(1.0 / etac) * normal_flux * geom.normals_z, t_rhs[idx_2d_rho_w, :end, :])



   return rhs
