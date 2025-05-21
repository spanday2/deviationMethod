import numpy
import pdb

from common.program_options     import Configuration
from common.definitions import idx_2d_rho, idx_2d_rho_u, idx_2d_rho_w, idx_2d_rho_theta,  \
                               p0, Rd, cpd, cvd, heat_capacity_ratio, gravity

def rhs_bubble(Q, geom, mtrx, nbsolpts, nb_elements_x, nb_elements_z):

   # theta_base                = numpy.ones_like(geom.X1)*303.15
   # exner_base                = (1.0 - gravity / (cpd * theta_base) * geom.X3)
   # rho_base                  = p0 / (Rd * theta_base) * exner_base**(cvd / Rd)
   # E_base                    = cvd*theta_base*exner_base + gravity*geom.X3    # We did not add 0.5*(u^2+w^2) because its zero
   #------------------------------------------------------------------------------------------------------------------------------
   gamma                     = 5/3
   c                         = 1 / (gamma - 1)
   g                         = 1
   ρ0                        = 1
   p0                        = 1
   rho_base                  = ρ0 * numpy.exp(- (ρ0/p0) * g * geom.X3)
   pressure_base             = p0 * numpy.exp(- (ρ0/p0) * g * geom.X3)
   E_base                    = c*(pressure_base / rho_base) + g*geom.X3
   Q_tilda                   = numpy.zeros_like(Q)
   Q_tilda[idx_2d_rho]       = rho_base
   Q_tilda[idx_2d_rho_theta] = rho_base * E_base
  

   Q_total = Q + Q_tilda


   def compute_rhs(Qv, geom, idx_2d_rho, idx_2d_rho_u, idx_2d_rho_w, idx_2d_rho_theta,  \
                     p0, Rd, cpd, cvd, heat_capacity_ratio, gravity):

      datatype = Qv.dtype
      nb_equations = Qv.shape[0] # Number of constituent Euler equations.  Probably 6.

      nb_interfaces_x = nb_elements_x + 1
      nb_interfaces_z = nb_elements_z + 1

      flux_x1, t_flux_x1 = [numpy.empty_like(Qv, dtype=datatype) for _ in range(2)]
      flux_x3, t_flux_x3 = [numpy.empty_like(Qv, dtype=datatype) for _ in range(2)]

      df1_dx1, t_df1_dx1 = [numpy.empty_like(Qv, dtype=datatype) for _ in range(2)]
      df3_dx3, t_df3_dx3 = [numpy.empty_like(Qv, dtype=datatype) for _ in range(2)]

      kfaces_flux, t_kfaces_flux = [numpy.empty((nb_equations, nb_elements_z, 2, nbsolpts*nb_elements_x), dtype=datatype) for _ in range(2)]
      kfaces_var, t_kfaces_var   = [numpy.empty((nb_equations, nb_elements_z, 2, nbsolpts*nb_elements_x), dtype=datatype) for _ in range(2)]
      kfaces_pres                = numpy.empty((nb_elements_z, 2, nbsolpts*nb_elements_x), dtype=datatype)

      ifaces_flux, t_ifaces_flux = [numpy.empty((nb_equations, nb_elements_x, nbsolpts*nb_elements_z, 2), dtype=datatype) for _ in range(2)]
      ifaces_var, t_ifaces_var   = [numpy.empty((nb_equations, nb_elements_x, nbsolpts*nb_elements_z, 2), dtype=datatype) for _ in range(2)]
      ifaces_pres                = numpy.empty((nb_elements_x, nbsolpts*nb_elements_z, 2), dtype=datatype)

      # --- Unpack physical variables
      rho      = Qv[idx_2d_rho,:,:]
      uu       = Qv[idx_2d_rho_u,:,:] / rho
      ww       = Qv[idx_2d_rho_w,:,:] / rho
      ee       = Qv[idx_2d_rho_theta,:,:] / rho
      height   = geom.X3

      pressure = (heat_capacity_ratio-1) * (Qv[idx_2d_rho_theta, :, :] - 0.5*rho*(uu**2 + ww**2) - rho*gravity*height)
      # pressure = (heat_capacity_ratio-1) * (Qv[idx_2d_rho_theta, :, :] - rho*gravity*height)
      

      # --- Compute the fluxes
      flux_x1[idx_2d_rho,:,:]       = Qv[idx_2d_rho_u,:,:]
      flux_x1[idx_2d_rho_u,:,:]     = Qv[idx_2d_rho_u,:,:] * uu + pressure
      flux_x1[idx_2d_rho_w,:,:]     = Qv[idx_2d_rho_u,:,:] * ww
      flux_x1[idx_2d_rho_theta,:,:] = (Qv[idx_2d_rho_theta,:,:] + pressure) * uu


      flux_x3[idx_2d_rho,:,:]       = Qv[idx_2d_rho_w,:,:]
      flux_x3[idx_2d_rho_u,:,:]     = Qv[idx_2d_rho_w,:,:] * uu
      flux_x3[idx_2d_rho_w,:,:]     = Qv[idx_2d_rho_w,:,:] * ww + pressure
      flux_x3[idx_2d_rho_theta,:,:] = (Qv[idx_2d_rho_theta,:,:] + pressure) * ww

      # --- Interpolate to the element interface
      standard_slice = numpy.arange(nbsolpts)
      for elem in range(nb_elements_z):
         epais = elem * nbsolpts + standard_slice

         kfaces_var[:,elem,0,:] = mtrx.extrap_down @ Qv[:,epais,:]
         kfaces_var[:,elem,1,:] = mtrx.extrap_up @ Qv[:,epais,:]
         kfaces_pres[elem,0,:]  = mtrx.extrap_down @ pressure[epais,:]
         kfaces_pres[elem,1,:]  = mtrx.extrap_up @ pressure[epais,:]

      for elem in range(nb_elements_x):
         epais = elem * nbsolpts + standard_slice

         ifaces_var[:,elem,:,0] = Qv[:,:,epais] @ mtrx.extrap_west
         ifaces_var[:,elem,:,1] = Qv[:,:,epais] @ mtrx.extrap_east
         ifaces_pres[elem,:,0]  = pressure[:,epais] @ mtrx.extrap_west
         ifaces_pres[elem,:,1]  = pressure[:,epais] @ mtrx.extrap_east

      # --- Bondary treatement

      # zeros flux BCs everywhere ...
      kfaces_flux[:,0,0,:]  = 0.0
      kfaces_flux[:,-1,1,:] = 0.0

      # # Apply Dirichlet boundaries in Z (top and bottom)
      # Q_bottom = numpy.zeros((4, nbsolpts * nb_elements_x), dtype=numpy.float64)
      # Q_bottom[0] = 1
      # Q_bottom[3] = 1 / ((5/3) - 1)
      # Q_top = numpy.zeros((4, nbsolpts * nb_elements_x), dtype=numpy.float64)
      # Q_top[0] = numpy.exp(-2)
      # Q_top[3] = (1 / ((5/3) - 1) + 2) * numpy.exp(-2)
      # rho_b = Q_bottom[idx_2d_rho]
      # rho_t = Q_top[idx_2d_rho]
      # u_b = Q_bottom[idx_2d_rho_u] / rho_b
      # w_b = Q_bottom[idx_2d_rho_w] / rho_b
      # e_b = Q_bottom[idx_2d_rho_theta] / rho_b
      # u_t = Q_top[idx_2d_rho_u] / rho_t
      # w_t = Q_top[idx_2d_rho_w] / rho_t
      # e_t = Q_top[idx_2d_rho_theta] / rho_t
      # h_b = 0
      # h_t = 2
      # p_b = (heat_capacity_ratio - 1) * (Q_bottom[idx_2d_rho_theta] - rho_b * gravity * h_b)
      # p_t = (heat_capacity_ratio - 1) * (Q_top[idx_2d_rho_theta] - rho_t * gravity * h_t)

      # kfaces_flux[idx_2d_rho, 0, 0, :]       = Q_bottom[idx_2d_rho_u]
      # kfaces_flux[idx_2d_rho_u, 0, 0, :]     = Q_bottom[idx_2d_rho_u] * w_b
      # kfaces_flux[idx_2d_rho_w, 0, 0, :]     = Q_bottom[idx_2d_rho_u] * w_b + p_b
      # kfaces_flux[idx_2d_rho_theta, 0, 0, :] = (Q_bottom[idx_2d_rho_theta] + p_b) * w_b

      # kfaces_flux[idx_2d_rho, -1, 1, :]       = Q_top[idx_2d_rho_u]
      # kfaces_flux[idx_2d_rho_u, -1, 1, :]     = Q_top[idx_2d_rho_u] * w_t 
      # kfaces_flux[idx_2d_rho_w, -1, 1, :]     = Q_top[idx_2d_rho_u] * w_t + p_t
      # kfaces_flux[idx_2d_rho_theta, -1, 1, :] = (Q_top[idx_2d_rho_theta, :] + p_t) * w_t


      # Skip periodic faces
      if not geom.xperiodic:
         ifaces_flux[:, 0,:,0] = 0.0
         ifaces_flux[:,-1,:,1] = 0.0

      # except for momentum eqs where pressure is extrapolated to BCs.
      kfaces_flux[idx_2d_rho_w, 0, 0, :] = kfaces_pres[ 0, 0, :]
      kfaces_flux[idx_2d_rho_w,-1, 1, :] = kfaces_pres[-1, 1, :]

      # ifaces_flux[idx_2d_rho_u, 0,:,0] = ifaces_pres[0,:,0]  # TODO : pour les cas théoriques seulement ...
      # ifaces_flux[idx_2d_rho_u,-1,:,1] = ifaces_pres[-1,:,1]

      # --- Common AUSM+ up fluxes
      for itf in range(1, nb_interfaces_z - 1):

         left  = itf - 1
         right = itf

         # # --- AUSM+up constants ---
         # sigma     = 1.0
         # K_p       = 0.25
         # K_u       = 0.75
         # beta      = 1.0 / 8.0 
         # M_inf_u   = 1
         # M_inf_p   = 1

         # # --- Primitive variables ---
         # rho_L = kfaces_var[idx_2d_rho, left, 1, :]
         # rho_R = kfaces_var[idx_2d_rho, right, 0, :]

         # u_L   = kfaces_var[idx_2d_rho_u, left, 1, :] / rho_L
         # u_R   = kfaces_var[idx_2d_rho_u, right, 0, :] / rho_R

         # w_L   = kfaces_var[idx_2d_rho_w, left, 1, :] / rho_L
         # w_R   = kfaces_var[idx_2d_rho_w, right, 0, :] / rho_R

         # e_L   = kfaces_var[idx_2d_rho_theta, left, 1, :] / rho_L
         # e_R   = kfaces_var[idx_2d_rho_theta, right, 0, :] / rho_R

         # p_L   = kfaces_pres[left, 1, :]
         # p_R   = kfaces_pres[right, 0, :]  

         # # Interface speed of sound
         # a_L   = numpy.sqrt(heat_capacity_ratio * p_L / rho_L)
         # a_R   = numpy.sqrt(heat_capacity_ratio * p_R / rho_R)
         # ahalf = 0.5*(a_L + a_R)  

         # # --- Compute interface Mach numbers ---
         # M_L     = w_L / ahalf
         # M_R     = w_R / ahalf

         # Mbar_sq = (w_L**2 + w_R**2) / (2.0 * ahalf**2)
         # Mo_sq   = numpy.minimum(1.0, numpy.maximum(Mbar_sq, M_inf_p**2))
         # Mo      = numpy.sqrt(Mo_sq)
         # fa      = Mo*(2-Mo) 

         # Mplus    = 0.25 * (M_L + 1)**2 * (1 + 16 * beta * 0.25 * (M_L - 1)**2)
         # Mminus   = -0.25 * (M_R - 1)**2 * (1 + 16 * beta * 0.25 * (M_R + 1)**2)

         # rhohalf  = 0.5 * (rho_L + rho_R)

         # # --- Mach number at interface (Eq. 73)
         # Mhalf = ( Mplus + Mminus - (K_p / fa) * numpy.maximum(1.0 - sigma * Mbar_sq, 0) * (p_R - p_L) / (rhohalf * ahalf**2))
      
         # # --- Compute mass flux (Eq. 74)
         # mdothalf = ahalf * Mhalf * numpy.where(Mhalf > 0, rho_L, rho_R)

         # # --- Compute pressure flux ---
         # Mo_sq   = numpy.minimum(1.0, numpy.maximum(Mbar_sq, M_inf_u**2))
         # Mo      = numpy.sqrt(Mo_sq)
         # fa      = Mo*(2-Mo) 
         # alpha = (3/16)*(-4 + 5*fa**2)

         # Pplus  = 0.25 * (M_L + 1)**2 * ((2 - M_L) + 16 * alpha * M_L * 0.25 * (M_L - 1)**2)
         # Pminus = -0.25 * (M_R - 1)**2 * ((-2 - M_R) + 16 * alpha * M_R * 0.25 * (M_R + 1)**2)


         # # --- Pressure flux (Eq. 75)
         # Phalf = Pplus * p_L + Pminus * p_R - K_u * Pplus * Pminus * (rho_L + rho_R) * ahalf * fa * (w_R - w_L)

         # # Boolean mask: True where mdothalf > 0
         # selector = mdothalf > 0
         # kfaces_flux[idx_2d_rho, right, 0, :] = mdothalf    # Mass flux (no condition needed, both branches use mdothalf)

         # # Apply conditional values using np.where
         # kfaces_flux[idx_2d_rho_u,     right, 0, :] = mdothalf * numpy.where(selector, u_L, u_R)
         # kfaces_flux[idx_2d_rho_w,     right, 0, :] = mdothalf * numpy.where(selector, w_L, w_R) + Phalf
         # kfaces_flux[idx_2d_rho_theta, right, 0, :] = mdothalf * numpy.where(selector, e_L, e_R) + (Phalf*ahalf*Mhalf)
 

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
                                      ((kfaces_var[3,right,0,:] + kfaces_pres[right,0,:]) * numpy.minimum(0., M) * a_R)

         kfaces_flux[idx_2d_rho_w,right,0,:] += 0.5 * ((1. + M_L) * kfaces_pres[left,1,:] + \
                                                      (1. - M_R) * kfaces_pres[right,0,:])

         kfaces_flux[:,left,1,:] = kfaces_flux[:,right,0,:]

      start = 0 if geom.xperiodic else 1
      for itf in range(start, nb_interfaces_x - 1):

         left  = itf - 1
         right = itf

         # # --- AUSM+up constants ---
         # sigma     = 1.0
         # K_p       = 0.25
         # K_u       = 0.75
         # beta      = 1.0 / 8.0 
         # M_inf_u   = 1 
         # M_inf_p   = 1

         # # --- Primitive variables ---
         # rho_L = ifaces_var[idx_2d_rho, left, :, 1]
         # rho_R = ifaces_var[idx_2d_rho, right, :, 0]

         # u_L   = ifaces_var[idx_2d_rho_u, left, :, 1] / rho_L
         # u_R   = ifaces_var[idx_2d_rho_u, right, :, 0] / rho_R

         # w_L   = ifaces_var[idx_2d_rho_w, left, :, 1] / rho_L
         # w_R   = ifaces_var[idx_2d_rho_w, right, :, 0] / rho_R

         # e_L   = ifaces_var[idx_2d_rho_theta, left, :, 1] / rho_L
         # e_R   = ifaces_var[idx_2d_rho_theta, right, :, 0] / rho_R

         # p_L   = ifaces_pres[left, :, 1]
         # p_R   = ifaces_pres[right, :, 0]  

         # # Interface speed of sound
         # a_L   = numpy.sqrt(heat_capacity_ratio * p_L / rho_L)
         # a_R   = numpy.sqrt(heat_capacity_ratio * p_R / rho_R)
         # ahalf = 0.5*(a_L + a_R)  

         # # --- Compute interface Mach numbers ---
         # M_L     = u_L / ahalf
         # M_R     = u_R / ahalf

         # Mbar_sq = (u_L**2 + u_R**2) / (2.0 * ahalf**2)
         # Mo_sq   = numpy.minimum(1.0, numpy.maximum(Mbar_sq, M_inf_p**2))
         # Mo      = numpy.sqrt(Mo_sq)
         # fa      = Mo*(2-Mo) 

         # Mplus    = 0.25 * (M_L + 1)**2 * (1 + 16 * beta * 0.25 * (M_L - 1)**2)
         # Mminus   = -0.25 * (M_R - 1)**2 * (1 + 16 * beta * 0.25 * (M_R + 1)**2)

         # rhohalf  = 0.5 * (rho_L + rho_R)

         # # --- Mach number at interface (Eq. 73)
         # Mhalf = ( Mplus + Mminus - (K_p / fa) * numpy.maximum(1.0 - sigma * Mbar_sq, 0) * (p_R - p_L) / (rhohalf * ahalf**2))
      
         # # --- Compute mass flux (Eq. 74)
         # mdothalf = ahalf * Mhalf * numpy.where(Mhalf > 0, rho_L, rho_R)

         # # --- Compute pressure flux ---
         # Mo_sq   = numpy.minimum(1.0, numpy.maximum(Mbar_sq, M_inf_u**2))
         # Mo      = numpy.sqrt(Mo_sq)
         # fa      = Mo*(2-Mo) 
         # alpha = (3/16)*(-4 + 5*fa**2)

         # Pplus  = 0.25 * (M_L + 1)**2 * ((2 - M_L) + 16 * alpha * M_L * 0.25 * (M_L - 1)**2)
         # Pminus = -0.25 * (M_R - 1)**2 * ((-2 - M_R) + 16 * alpha * M_R * 0.25 * (M_R + 1)**2)


         # # --- Pressure flux (Eq. 75)
         # Phalf = Pplus * p_L + Pminus * p_R - K_u * Pplus * Pminus * (rho_L + rho_R) * ahalf * fa * (u_R - u_L)

         # # Boolean mask: True where mdothalf > 0
         # selector = mdothalf > 0
         # ifaces_flux[idx_2d_rho, right, :, 0] = mdothalf    # Mass flux (no condition needed, both branches use mdothalf)

         # # Apply conditional values using np.where
         # ifaces_flux[idx_2d_rho_u,     right, :, 0] = mdothalf * numpy.where(selector, u_L, u_R) + Phalf
         # ifaces_flux[idx_2d_rho_w,     right, :, 0] = mdothalf * numpy.where(selector, w_L, w_R) 
         # ifaces_flux[idx_2d_rho_theta, right, :, 0] = mdothalf * numpy.where(selector, e_L, e_R) + (Phalf*ahalf*Mhalf)
 

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

      rhs[idx_2d_rho_w,:,:] -= Qv[idx_2d_rho,:,:] * gravity

      return rhs

   rhs   = compute_rhs(Q_total, geom, idx_2d_rho, idx_2d_rho_u, idx_2d_rho_w, idx_2d_rho_theta,  \
                        p0, Rd, cpd, cvd, heat_capacity_ratio, gravity)

   t_rhs = compute_rhs(Q_tilda, geom, idx_2d_rho, idx_2d_rho_u, idx_2d_rho_w, idx_2d_rho_theta,  \
                        p0, Rd, cpd, cvd, heat_capacity_ratio, gravity)


   # return rhs
   return rhs - t_rhs

      