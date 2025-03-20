import numpy
import pdb

from common.program_options     import Configuration
from common.definitions import idx_2d_rho, idx_2d_rho_u, idx_2d_rho_w, idx_2d_rho_theta,  \
                               p0, Rd, cpd, cvd, heat_capacity_ratio, gravity

def rhs_bubble(Q, geom, mtrx, nbsolpts, nb_elements_x, nb_elements_z):

   theta_base                = numpy.ones_like(geom.X1)*303.15
   exner_base                = (1.0 - gravity / (cpd * theta_base) * geom.X3)
   rho_base                  = p0 / (Rd * theta_base) * exner_base**(cvd / Rd)
   E_base                    = cvd*theta_base*exner_base + gravity*geom.X3    # We did not add 0.5*(u^2+w^2) because its zero
   Q_tilda                   = numpy.zeros_like(Q)
   Q_tilda[idx_2d_rho]       = rho_base
   Q_tilda[idx_2d_rho_theta] = rho_base * E_base


   Q_total = Q + Q_tilda

   def compute_rhs(Q, geom, idx_2d_rho, idx_2d_rho_u, idx_2d_rho_w, idx_2d_rho_theta,  \
                     p0, Rd, cpd, cvd, heat_capacity_ratio, gravity):

      datatype = Q.dtype
      nb_equations = Q.shape[0] # Number of constituent Euler equations.  Probably 6.

      nb_interfaces_x = nb_elements_x + 1
      nb_interfaces_z = nb_elements_z + 1

      flux_x1, t_flux_x1 = [numpy.empty_like(Q, dtype=datatype) for _ in range(2)]
      flux_x3, t_flux_x3 = [numpy.empty_like(Q, dtype=datatype) for _ in range(2)]

      df1_dx1, t_df1_dx1 = [numpy.empty_like(Q, dtype=datatype) for _ in range(2)]
      df3_dx3, t_df3_dx3 = [numpy.empty_like(Q, dtype=datatype) for _ in range(2)]

      kfaces_flux, t_kfaces_flux = [numpy.empty((nb_equations, nb_elements_z, 2, nbsolpts*nb_elements_x), dtype=datatype) for _ in range(2)]
      kfaces_var, t_kfaces_var   = [numpy.empty((nb_equations, nb_elements_z, 2, nbsolpts*nb_elements_x), dtype=datatype) for _ in range(2)]
      kfaces_pres                = numpy.empty((nb_elements_z, 2, nbsolpts*nb_elements_x), dtype=datatype)

      ifaces_flux, t_ifaces_flux = [numpy.empty((nb_equations, nb_elements_x, nbsolpts*nb_elements_z, 2), dtype=datatype) for _ in range(2)]
      ifaces_var, t_ifaces_var   = [numpy.empty((nb_equations, nb_elements_x, nbsolpts*nb_elements_z, 2), dtype=datatype) for _ in range(2)]
      ifaces_pres                = numpy.empty((nb_elements_x, nbsolpts*nb_elements_z, 2), dtype=datatype)


      # --- Unpack physical variables
      rho      = Q[idx_2d_rho,:,:]
      uu       = Q[idx_2d_rho_u,:,:] / rho
      ww       = Q[idx_2d_rho_w,:,:] / rho
      height   = geom.X3

      pressure = (heat_capacity_ratio-1) * (Q[idx_2d_rho_theta, :, :] - 0.5*rho*(uu**2 + ww**2) - rho*gravity*height)


      # --- Compute the fluxes
      flux_x1[idx_2d_rho,:,:]       = Q[idx_2d_rho_u,:,:]
      flux_x1[idx_2d_rho_u,:,:]     = Q[idx_2d_rho_u,:,:] * uu + pressure
      flux_x1[idx_2d_rho_w,:,:]     = Q[idx_2d_rho_u,:,:] * ww
      flux_x1[idx_2d_rho_theta,:,:] = (Q[idx_2d_rho_theta,:,:] + pressure) * uu

      flux_x3[idx_2d_rho,:,:]       = Q[idx_2d_rho_w,:,:]
      flux_x3[idx_2d_rho_u,:,:]     = Q[idx_2d_rho_w,:,:] * uu
      flux_x3[idx_2d_rho_w,:,:]     = Q[idx_2d_rho_w,:,:] * ww + pressure
      flux_x3[idx_2d_rho_theta,:,:] = (Q[idx_2d_rho_theta,:,:] + pressure) * ww

      # --- Interpolate to the element interface
      standard_slice = numpy.arange(nbsolpts)
      for elem in range(nb_elements_z):
         epais = elem * nbsolpts + standard_slice

         kfaces_var[:,elem,0,:] = mtrx.extrap_down @ Q[:,epais,:]
         kfaces_var[:,elem,1,:] = mtrx.extrap_up @ Q[:,epais,:]
         kfaces_pres[elem,0,:]  = mtrx.extrap_down @ pressure[epais,:]
         kfaces_pres[elem,1,:]  = mtrx.extrap_up @ pressure[epais,:]

      for elem in range(nb_elements_x):
         epais = elem * nbsolpts + standard_slice

         ifaces_var[:,elem,:,0] = Q[:,:,epais] @ mtrx.extrap_west
         ifaces_var[:,elem,:,1] = Q[:,:,epais] @ mtrx.extrap_east
         ifaces_pres[elem,:,0]  = pressure[:,epais] @ mtrx.extrap_west
         ifaces_pres[elem,:,1]  = pressure[:,epais] @ mtrx.extrap_east


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

         kfaces_flux[0:3,right,0,:] = (kfaces_var[0:3,left,1,:] * numpy.maximum(0., M) * a_L) + \
                                      (kfaces_var[0:3,right,0,:] * numpy.minimum(0., M) * a_R)
         kfaces_flux[3,right,0,:]   = ((kfaces_var[3,left,1,:] + kfaces_pres[left,1,:]) * numpy.maximum(0., M) * a_L) + \
                                      ((kfaces_var[3,right,0,:] +  kfaces_pres[right,0,:]) * numpy.minimum(0., M) * a_R)


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

         ifaces_flux[0:3,right,:,0] = (ifaces_var[0:3,left,:,1] * numpy.maximum(0., M) * a_L) + \
                                      (ifaces_var[0:3,right,:,0] * numpy.minimum(0., M) * a_R)
         ifaces_flux[3,right,:,0]   = ((ifaces_var[3,left,:,1] + ifaces_pres[left,:,1]) * numpy.maximum(0., M) * a_L) + \
                                      ((ifaces_var[3,right,:,0] + ifaces_pres[right,:,1]) * numpy.minimum(0., M) * a_R)


         ifaces_flux[idx_2d_rho_u,right,:,0] += 0.5 * ((1. + M_L) * ifaces_pres[left,:,1] + \
                                                      (1. - M_R) * ifaces_pres[right,:,0])

         ifaces_flux[:,left,:,1] = ifaces_flux[:,right,:,0]

      if geom.xperiodic:
         ifaces_flux[:, 0, :, 0] = ifaces_flux[:, -1, :, 1]

      # --- Compute the derivatives
      for elem in range(nb_elements_z):
         epais = elem * nbsolpts + standard_slice
         factor = 2.0 / geom.Δx3

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

   rhs   = compute_rhs(Q_total, geom, idx_2d_rho, idx_2d_rho_u, idx_2d_rho_w, idx_2d_rho_theta,  \
                        p0, Rd, cpd, cvd, heat_capacity_ratio, gravity)

   t_rhs = compute_rhs(Q_tilda, geom, idx_2d_rho, idx_2d_rho_u, idx_2d_rho_w, idx_2d_rho_theta,  \
                        p0, Rd, cpd, cvd, heat_capacity_ratio, gravity)

   return rhs - t_rhs

      