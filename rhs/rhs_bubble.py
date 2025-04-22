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
   gamma                     = heat_capacity_ratio
   # gamma                     = 1.4
   # c                         = 1 / (gamma - 1)
   # g                         = 1
   # ρ0                        = 1
   # rho_base                  = ρ0 * numpy.exp(- (ρ0/p0) * g * geom.X3)
   # pressure_base             = p0 * numpy.exp(- (ρ0/p0) * g * geom.X3)
   # E_base                    = c*(pressure_base / rho_base) + g*geom.X3
   Q_tilda                   = numpy.zeros_like(Q)
   Q_tilda[idx_2d_rho]       = rho_base
   Q_tilda[idx_2d_rho_theta] = rho_base * E_base


   Q_total = Q + Q_tilda

   #-----------------------------------------------------------------------------------------------#

   # --- Mach splitting functions (Eq. 20) ---
   def M4_plus(M, beta):
      absM = numpy.abs(M)
      M2_plus = 0.25 * (M + 1)**2
      M2_minus = -0.25 * (M - 1)**2
      return numpy.where(
            absM >= 1,
            0.5 * (M + absM),
            M2_plus * (1 - 16 * beta * M2_minus)
      )

   def M4_minus(M, beta):
      absM = numpy.abs(M)
      M2_plus = 0.25 * (M + 1)**2
      M2_minus = -0.25 * (M - 1)**2
      return numpy.where(
            absM >= 1,
            0.5 * (M - absM),
            M2_minus * (1 + 16 * beta * M2_plus)
      )

   def P5_plus(M, alpha):
      absM = numpy.abs(M)
      M1_plus = 0.5 * (M + absM)
      M2_plus = 0.25 * (M + 1)**2
      M2_minus = -0.25 * (M - 1)**2
      return numpy.where(
            absM >= 1,
            M1_plus / M,
            M2_plus * ((2 - M) - 16 * alpha * M * M2_minus)
      )

   def P5_minus(M, alpha):
      absM = numpy.abs(M)
      M1_minus = 0.5 * (M - absM)
      M2_plus = 0.25 * (M + 1)**2
      M2_minus = -0.25 * (M - 1)**2
      return numpy.where(
            absM >= 1,
            M1_minus / M,
            M2_minus * ((-2 - M) + 16 * alpha * M * M2_plus)
      )

   def compute_rhs(Qv, geom, idx_2d_rho, idx_2d_rho_u, idx_2d_rho_w, idx_2d_rho_theta,  \
                     p0, Rd, cpd, cvd, gamma, gravity):

      datatype = Qv.dtype
      nb_equations = Qv.shape[0] # Number of constituent Euler equations.  Probably 6.


      nb_interfaces_x = nb_elements_x + 1
      nb_interfaces_z = nb_elements_z + 1

      flux_x1 = numpy.empty_like(Qv, dtype=datatype)
      flux_x3 = numpy.empty_like(Qv, dtype=datatype)

      df1_dx1 = numpy.empty_like(Qv, dtype=datatype)
      df3_dx3 = numpy.empty_like(Qv, dtype=datatype)

      kfaces_flux                = numpy.empty((nb_equations, nb_elements_z, 2, nbsolpts*nb_elements_x), dtype=datatype)
      kfaces_var                 = numpy.empty((nb_equations, nb_elements_z, 2, nbsolpts*nb_elements_x), dtype=datatype)
      kfaces_pres                = numpy.empty((nb_elements_z, 2, nbsolpts*nb_elements_x), dtype=datatype)
      kfaces_enthalpy            = numpy.empty((nb_elements_z, 2, nbsolpts*nb_elements_x), dtype=datatype)

      ifaces_flux                = numpy.empty((nb_equations, nb_elements_x, nbsolpts*nb_elements_z, 2), dtype=datatype)
      ifaces_var                 = numpy.empty((nb_equations, nb_elements_x, nbsolpts*nb_elements_z, 2), dtype=datatype)
      ifaces_pres                = numpy.empty((nb_elements_x, nbsolpts*nb_elements_z, 2), dtype=datatype)
      ifaces_enthalpy            = numpy.empty((nb_elements_x, nbsolpts*nb_elements_z, 2), dtype=datatype)
 


      # --- Unpack physical variables
      rho      = Qv[idx_2d_rho,:,:]
      uu       = Qv[idx_2d_rho_u,:,:] / rho
      ww       = Qv[idx_2d_rho_w,:,:] / rho
      ee       = Qv[idx_2d_rho_theta,:,:] / rho
      height   = geom.X3


      pressure = (gamma-1) * (Qv[idx_2d_rho_theta, :, :] - 0.5*rho*(uu**2 + ww**2) - rho*gravity*height)
      enthalpy = (gamma/(gamma-1))*(pressure/rho) + 0.5*(uu**2 + ww**2) + gravity*height
      enthalpy2 = ee + (pressure/rho)
 
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

         kfaces_var[:,elem,0,:]    = mtrx.extrap_down @ Qv[:,epais,:]
         kfaces_var[:,elem,1,:]    = mtrx.extrap_up @ Qv[:,epais,:]
         kfaces_pres[elem,0,:]     = mtrx.extrap_down @ pressure[epais,:]
         kfaces_pres[elem,1,:]     = mtrx.extrap_up @ pressure[epais,:]
         kfaces_enthalpy[elem,0,:] = mtrx.extrap_down @ enthalpy[epais,:]
         kfaces_enthalpy[elem,1,:] = mtrx.extrap_up @ enthalpy[epais,:]

      for elem in range(nb_elements_x):
         epais = elem * nbsolpts + standard_slice

         ifaces_var[:,elem,:,0]    = Qv[:,:,epais] @ mtrx.extrap_west
         ifaces_var[:,elem,:,1]    = Qv[:,:,epais] @ mtrx.extrap_east
         ifaces_pres[elem,:,0]     = pressure[:,epais] @ mtrx.extrap_west
         ifaces_pres[elem,:,1]     = pressure[:,epais] @ mtrx.extrap_east
         ifaces_enthalpy[elem,:,0] = enthalpy[:,epais] @ mtrx.extrap_west
         ifaces_enthalpy[elem,:,1] = enthalpy[:,epais] @ mtrx.extrap_east


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
   
      
      # --- Common AUSM+ UP fluxes (vertical direction)
      # start = 0 if geom.zperiodic else 1
      
      for itf in range(1, nb_interfaces_z - 1):
         left  = itf - 1
         right = itf

         # --- AUSM+up constants ---
         sigma     = 1.0
         K_p       = 0.25
         K_u       = 0.75
         beta      = 1.0 / 8.0 
         M_inf     = 1e-13 

         # --- Primitive variables ---
         rho_L = kfaces_var[idx_2d_rho, left, 1, :]
         rho_R = kfaces_var[idx_2d_rho, right, 0, :]

         u_L   = kfaces_var[idx_2d_rho_u, left, 1, :] / rho_L
         u_R   = kfaces_var[idx_2d_rho_u, right, 0, :] / rho_R

         w_L   = kfaces_var[idx_2d_rho_w, left, 1, :] / rho_L
         w_R   = kfaces_var[idx_2d_rho_w, right, 0, :] / rho_R

         H_L   = kfaces_enthalpy[left, 1, :]
         H_R   = kfaces_enthalpy[right, 0, :]

         p_L   = kfaces_pres[left, 1, :]
         p_R   = kfaces_pres[right, 0, :]    

         # --- Interface speed of sound ---
   
         a_L   = numpy.sqrt(heat_capacity_ratio * p_L / rho_L)
         a_R   = numpy.sqrt(heat_capacity_ratio * p_R / rho_R)
         ahalf = 0.5*(a_L + a_R)

         # astarsqr_L = (2 * (gamma - 1.) * H_L) / (gamma + 1)  
         # astarsqr_R = (2 * (gamma - 1.) * H_R) / (gamma + 1) 

         # astar_L    = numpy.sqrt( astarsqr_L )
         # astar_R    = numpy.sqrt( astarsqr_R )
         # pdb.set_trace()
         # ahat_L     = astarsqr_L / ( numpy.maximum( astar_L, numpy.abs( w_L ) ) ) 
         # ahat_R     = astarsqr_R / ( numpy.maximum( astar_R, numpy.abs( w_R ) ) )

         # ahalf      = numpy.minimum( ahat_L, ahat_R )

         # --- Compute interface Mach numbers ---
         M_L     = w_L / ahalf
         M_R     = w_R / ahalf

         Mbar_sq = (w_L**2 + w_R**2) / (2.0 * ahalf**2)
         Mo_sq   = numpy.minimum(1.0, numpy.maximum(Mbar_sq, M_inf**2))
         Mo      = numpy.sqrt(Mo_sq)
         fa      = Mo*(2-Mo) 

         Mplus    = M4_plus(M_L, beta)   # M+ = M4+(M_L)
         Mminus   = M4_minus(M_R, beta)  # M- = M4-(M_R)

         rhohalf  = 0.5 * (rho_L + rho_R)

         # --- Mach number at interface (Eq. 73)
         Mhalf = ( Mplus + Mminus - (K_p / fa) * numpy.maximum(1.0 - sigma * Mbar_sq, 0) * (p_R - p_L) / (rhohalf * ahalf**2))
        
         # --- Compute mass flux (Eq. 74)
         mdothalf = ahalf * Mhalf * numpy.where(Mhalf > 0, rho_L, rho_R)

         # --- Compute pressure flux ---
         alpha = (3/16)*(-4 + 5*fa**2)

         Pplus  = P5_plus(M_L, alpha)
         Pminus = P5_minus(M_R, alpha)

         # --- Pressure flux (Eq. 75)
         Phalf = Pplus * p_L + Pminus * p_R - K_u * Pplus * Pminus * (rho_L + rho_R) * ahalf * fa * (w_R - w_L)

         # Boolean mask: True where mdothalf > 0
         selector = mdothalf > 0
         kfaces_flux[idx_2d_rho, right, 0, :] = mdothalf    # Mass flux (no condition needed, both branches use mdothalf)

         # Apply conditional values using np.where
         kfaces_flux[idx_2d_rho_u,     right, 0, :] = mdothalf * numpy.where(selector, u_L, u_R)
         kfaces_flux[idx_2d_rho_w,     right, 0, :] = mdothalf * numpy.where(selector, w_L, w_R) + Phalf
         kfaces_flux[idx_2d_rho_theta, right, 0, :] = mdothalf * numpy.where(selector, H_L, H_R)
         
         # --- Symmetric update for left face
         kfaces_flux[:, left, 1, :] = kfaces_flux[:, right, 0, :]



      # if geom.zperiodic:
      #    kfaces_flux[:, 0, 0, :] = kfaces_flux[:, -1, 1, :]
      # ifaces flux
      start      = 0 if geom.xperiodic else 1
      for itf in range(start, nb_interfaces_x - 1):
   
         left    = itf - 1
         right   = itf

         # Primitive variables
         rho_L = ifaces_var[idx_2d_rho, left, :, 1]
         rho_R = ifaces_var[idx_2d_rho, right, :, 0]

         u_L   = ifaces_var[idx_2d_rho_u, left, :, 1] / rho_L
         u_R   = ifaces_var[idx_2d_rho_u, right, :, 0] / rho_R

         w_L   = ifaces_var[idx_2d_rho_w, left, :, 1] / rho_L
         w_R   = ifaces_var[idx_2d_rho_w, right, :, 0] / rho_R

         H_L   = ifaces_enthalpy[left, :, 1]
         H_R   = ifaces_enthalpy[right, :, 0]

         p_L   = ifaces_pres[left, :, 1]
         p_R   = ifaces_pres[right, :, 0]

         # Speed of sound at interface
         a_L = numpy.sqrt(heat_capacity_ratio * p_L / rho_L)
         a_R = numpy.sqrt(heat_capacity_ratio * p_R / rho_R)
         pdb.set_trace()
         ahalf = 0.5*(a_L + a_R)

         # astar2_L = 2 * (gamma - 1.) * H_L / (gamma + 1)
         # astar2_R = 2 * (gamma - 1.) * H_R / (gamma + 1)

         # astar_L = numpy.sqrt(astar2_L)
         # astar_R = numpy.sqrt(astar2_R)

         # ahat_L  = astar2_L / numpy.maximum(astar_L, numpy.abs(u_L))
         # ahat_R  = astar2_R / numpy.maximum(astar_R, numpy.abs(u_R))

         # ahat    = numpy.minimum(ahat_L, ahat_R)

         # Mach numbers
         M_L = u_L / ahalf
         M_R = u_R / ahalf

         Mbar_sq = 0.5 * (u_L**2 + u_R**2) / ahalf**2
         Mo_sq   = numpy.minimum(1.0, numpy.maximum(Mbar_sq, M_inf**2))
         Mo      = numpy.sqrt(Mo_sq)
         fa      = Mo*(2-Mo) 


         Mplus  = M4_plus(M_L, beta)
         Mminus = M4_minus(M_R, beta)

         rhohalf = 0.5 * (rho_L + rho_R)

         # Interface Mach number (Eq. 73)
         Mhalf = (Mplus + Mminus - (K_p / fa) * numpy.maximum(1 - sigma * Mbar_sq, 0) * (p_R - p_L) / (rhohalf * ahalf**2))

         # Mass flux
         mdothalf = ahalf * Mhalf * numpy.where(Mhalf > 0, rho_L, rho_R)

         # Pressure flux
         alpha = (3/16)*(-4 + 5*fa**2)

         Pplus  = P5_plus(M_L, alpha)
         Pminus = P5_minus(M_R, alpha)

         Phalf = Pplus * p_L + Pminus * p_R - K_u * Pplus * Pminus * (rho_L + rho_R) * ahalf * fa * (u_R - u_L)

         # AUSM+–up Flux Assembly
         selector = mdothalf > 0
         ifaces_flux[idx_2d_rho, right, :, 0]       = mdothalf
         ifaces_flux[idx_2d_rho_u, right, :, 0]     = mdothalf * numpy.where(selector, u_L, u_R) + Phalf
         ifaces_flux[idx_2d_rho_w, right, :, 0]     = mdothalf * numpy.where(selector, w_L, w_R)
         ifaces_flux[idx_2d_rho_theta, right, :, 0] = mdothalf * numpy.where(selector, H_L, H_R)

         # Symmetric update for left face
         ifaces_flux[:, left, :, 1] = ifaces_flux[:, right, :, 0]

   
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
                        p0, Rd, cpd, cvd, gamma, gravity)

   t_rhs = compute_rhs(Q_tilda, geom, idx_2d_rho, idx_2d_rho_u, idx_2d_rho_w, idx_2d_rho_theta,  \
                        p0, Rd, cpd, cvd, gamma, gravity)
   
   

   return rhs - t_rhs

      