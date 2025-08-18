import numpy
import pdb

from common.program_options     import Configuration
from common.definitions import idx_2d_rho, idx_2d_rho_u, idx_2d_rho_w, idx_2d_rho_theta,  \
                               p0, Rd, cpd, cvd, heat_capacity_ratio, gravity
from .flux import roe_flux_2d_delta0

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

      flux_x1 = numpy.empty_like(Qv, dtype=datatype)
      flux_x3 = numpy.empty_like(Qv, dtype=datatype)

      df1_dx1 = numpy.empty_like(Qv, dtype=datatype)
      df3_dx3 = numpy.empty_like(Qv, dtype=datatype)

      kfaces_flux     = numpy.zeros((nb_equations, nb_elements_z, 2, nbsolpts*nb_elements_x), dtype=datatype)
      kfaces_var      = numpy.zeros((nb_equations, nb_elements_z, 2, nbsolpts*nb_elements_x), dtype=datatype)
      kfaces_pres     = numpy.zeros((nb_elements_z, 2, nbsolpts*nb_elements_x), dtype=datatype)
      kfaces_enthalpy = numpy.zeros((nb_elements_z, 2, nbsolpts*nb_elements_x), dtype=datatype)
      kfaces_height   = numpy.zeros((nb_elements_z, 2, nbsolpts*nb_elements_x), dtype=datatype)

      ifaces_flux     = numpy.zeros((nb_equations, nb_elements_x, nbsolpts*nb_elements_z, 2), dtype=datatype)
      ifaces_var      = numpy.zeros((nb_equations, nb_elements_x, nbsolpts*nb_elements_z, 2), dtype=datatype)
      ifaces_pres     = numpy.zeros((nb_elements_x, nbsolpts*nb_elements_z, 2), dtype=datatype)
      ifaces_enthalpy = numpy.zeros((nb_elements_x, nbsolpts*nb_elements_z, 2), dtype=datatype)
      ifaces_height   = numpy.zeros((nb_elements_x, nbsolpts*nb_elements_z, 2), dtype=datatype)


      # --- Unpack physical variables
      rho      = Qv[idx_2d_rho,:,:]
      uu       = Qv[idx_2d_rho_u,:,:] / rho
      ww       = Qv[idx_2d_rho_w,:,:] / rho
      ee       = Qv[idx_2d_rho_theta,:,:] / rho
      height   = geom.X3

      pressure = (heat_capacity_ratio-1) * (Qv[idx_2d_rho_theta, :, :] - 0.5*rho*(uu**2 + ww**2) - rho*gravity*geom.X3)
      enthalpy = (heat_capacity_ratio/(heat_capacity_ratio-1))*(pressure/rho) + 0.5*(uu**2 + ww**2) + gravity*geom.X3

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

         kfaces_var[:,elem,0,:]    = (mtrx.extrap_down @ Qv[:,epais,:])
         kfaces_var[:,elem,1,:]    = (mtrx.extrap_up   @ Qv[:,epais,:])
         kfaces_pres[elem,0,:]     = (mtrx.extrap_down @ pressure[epais,:])
         kfaces_pres[elem,1,:]     = (mtrx.extrap_up   @ pressure[epais,:])
         kfaces_height[elem,0,:]   = (mtrx.extrap_down @ height[epais,:])
         kfaces_height[elem,1,:]   = (mtrx.extrap_up   @ height[epais,:])
         kfaces_enthalpy[elem,0,:] = (mtrx.extrap_down @ enthalpy[epais,:])
         kfaces_enthalpy[elem,1,:] = (mtrx.extrap_up   @ enthalpy[epais,:])


      for elem in range(nb_elements_x):
         epais = elem * nbsolpts + standard_slice

         ifaces_var[:,elem,:,0]    = (Qv[:,:,epais]   @ mtrx.extrap_west)
         ifaces_var[:,elem,:,1]    = (Qv[:,:,epais]   @ mtrx.extrap_east)
         ifaces_pres[elem,:,0]     = (pressure[:,epais] @ mtrx.extrap_west)
         ifaces_pres[elem,:,1]     = (pressure[:,epais] @ mtrx.extrap_east)
         ifaces_height[elem,:,0]   = (height[:,epais]   @ mtrx.extrap_west)
         ifaces_height[elem,:,1]   = (height[:,epais]   @ mtrx.extrap_east)
         ifaces_enthalpy[elem,:,0] = (enthalpy[:,epais] @ mtrx.extrap_west)
         ifaces_enthalpy[elem,:,1] = (enthalpy[:,epais] @ mtrx.extrap_east)

      # --- Bondary treatement

      # # zeros flux BCs everywhere ...
      # kfaces_flux[:,0,0,:]  = 0.0
      # kfaces_flux[:,-1,1,:] = 0.0

      # # except for momentum eqs where pressure is extrapolated to BCs.
      # kfaces_flux[idx_2d_rho_w, 0, 0, :] = kfaces_pres[ 0, 0, :]
      # kfaces_flux[idx_2d_rho_w,-1, 1, :] = kfaces_pres[-1, 1, :]

      UB_right_var       = numpy.zeros((nb_equations,nbsolpts*nb_elements_x)) # Free stream values at the upper boundary

      UB_right_var[idx_2d_rho]          = numpy.exp(-2) # From Background profile
      UB_right_var[idx_2d_rho_u]        = 0 #kfaces_var[idx_2d_rho_u,-1,1,:]
      UB_right_var[idx_2d_rho_w]        = 0 #-kfaces_var[idx_2d_rho_w,-1,1,:]
      UB_right_var[idx_2d_rho_theta]    = numpy.exp(-2) * (1/(heat_capacity_ratio-1) + gravity*2) # From Background profile
      UB_right_enthalpy = (heat_capacity_ratio/(heat_capacity_ratio-1)) + gravity*2 # From Background profile
 
      kfaces_u   = kfaces_var[idx_2d_rho_u,-1,1,:] / kfaces_var[idx_2d_rho,-1,1,:]
      kfaces_w   = kfaces_var[idx_2d_rho_w,-1,1,:] / kfaces_var[idx_2d_rho,-1,1,:]
      
      # Common flux at the top boundary
      kfaces_flux[:,-1,1,:] = roe_flux_2d_delta0(
            kfaces_var[:,-1,1,:], UB_right_var,
            u_L=kfaces_u, u_R=UB_right_var[idx_2d_rho_u]/UB_right_var[idx_2d_rho],
            w_L=kfaces_w, w_R=UB_right_var[idx_2d_rho_w]/UB_right_var[idx_2d_rho],
            p_L=kfaces_pres[-1,  1, :], p_R=numpy.exp(-2),
            H_L=kfaces_enthalpy[-1,  1, :], H_R=UB_right_enthalpy,
            h_L=kfaces_height[-1,1,:],   h_R=2,
            gamma=heat_capacity_ratio, gravity=gravity, normal="z")
      
  
      LB_left_var       = numpy.zeros((nb_equations,nbsolpts*nb_elements_x)) # Free stream values at the lower boundary

      LB_left_var[idx_2d_rho]          = 1 # From Background profile
      LB_left_var[idx_2d_rho_u]        = 0 #kfaces_var[idx_2d_rho_u,0,0,:]
      LB_left_var[idx_2d_rho_w]        = 0 #-kfaces_var[idx_2d_rho_w,0,0,:]
      LB_left_var[idx_2d_rho_theta]    = 1 * (1/(heat_capacity_ratio-1)) # From Background profile
      LB_left_enthalpy = (heat_capacity_ratio/(heat_capacity_ratio-1))   # From Background profile
      
      kfaces_u   = kfaces_var[idx_2d_rho_u,0,0,:] / kfaces_var[idx_2d_rho,0,0,:]
      kfaces_w   = kfaces_var[idx_2d_rho_w,0,0,:] / kfaces_var[idx_2d_rho,0,0,:]
      
      # Common flux at the bottom boundary
      kfaces_flux[:,0,0,:] = roe_flux_2d_delta0(
            LB_left_var, kfaces_var[:,0,0,:],
            u_L=LB_left_var[idx_2d_rho_u]/LB_left_var[idx_2d_rho], u_R=kfaces_u,
            w_L=LB_left_var[idx_2d_rho_w]/LB_left_var[idx_2d_rho], w_R=kfaces_w,
            p_L=1, p_R=kfaces_pres[0, 0, :],
            H_L=LB_left_enthalpy, H_R=kfaces_enthalpy[0, 0, :],
            h_L=0,   h_R=kfaces_height[0,0,:],
            gamma=heat_capacity_ratio, gravity=gravity, normal="z"
         )

      
      # Skip periodic faces
      if not geom.xperiodic:
         ifaces_flux[:, 0,:,0] = 0.0
         ifaces_flux[:,-1,:,1] = 0.0

      # ifaces_flux[idx_2d_rho_u, 0,:,0] = ifaces_pres[0,:,0]  # TODO : pour les cas théoriques seulement ...
      # ifaces_flux[idx_2d_rho_u,-1,:,1] = ifaces_pres[-1,:,1]

      # Compute ifaces u and w
      ifaces_u   = ifaces_var[idx_2d_rho_u] / ifaces_var[idx_2d_rho]
      ifaces_w   = ifaces_var[idx_2d_rho_w] / ifaces_var[idx_2d_rho] 
      kfaces_u   = kfaces_var[idx_2d_rho_u] / kfaces_var[idx_2d_rho]
      kfaces_w   = kfaces_var[idx_2d_rho_w] / kfaces_var[idx_2d_rho]

      
      # --- Common Roe fluxes
      # start = 0 if geom.zperiodic else 1
      start = 1
      for itf in range(start, nb_interfaces_z - 1):

         left  = itf - 1
         right = itf

         
         # Gather left/right slices (shape (neq, N))
         UL = kfaces_var[:, left,  1, :]
         UR = kfaces_var[:, right, 0, :]

         flux = roe_flux_2d_delta0(
            UL, UR,
            u_L=kfaces_u[left,  1, :], u_R=kfaces_u[right, 0, :],
            w_L=kfaces_w[left,  1, :], w_R=kfaces_w[right, 0, :],
            p_L=kfaces_pres[left,  1, :], p_R=kfaces_pres[right, 0, :],
            H_L=kfaces_enthalpy[left,  1, :], H_R=kfaces_enthalpy[right, 0, :],
            h_L=kfaces_height[left,  1, :],   h_R=kfaces_height[right, 0, :],
            gamma=heat_capacity_ratio, gravity=gravity, normal="z"
         )

         kfaces_flux[:, right, 0, :] = flux
         kfaces_flux[:, left,  1, :] = flux  # mirror


      # if geom.zperiodic:
      #    kfaces_flux[:, 0, 0, :] = kfaces_flux[:, -1, 1, :]



      # ifaces flux
      start      = 0 if geom.xperiodic else 1
      for itf in range(start, nb_interfaces_x - 1):

         left    = itf - 1
         right   = itf

         # Slices (neq, N)
         UL = ifaces_var[:, left,  :, 1]
         UR = ifaces_var[:, right, :, 0]

         flux = roe_flux_2d_delta0(
            UL, UR,
            u_L=ifaces_u[left,  :, 1], u_R=ifaces_u[right, :, 0],
            w_L=ifaces_w[left,  :, 1], w_R=ifaces_w[right, :, 0],
            p_L=ifaces_pres[left,  :, 1], p_R=ifaces_pres[right, :, 0],
            H_L=ifaces_enthalpy[left,  :, 1], H_R=ifaces_enthalpy[right, :, 0],
            h_L=ifaces_height[left,  :, 1],   h_R=ifaces_height[right, :, 0],
            gamma=heat_capacity_ratio, gravity=gravity, normal="x"
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

      
      return rhs

   rhs   = compute_rhs(Q_total, geom, idx_2d_rho, idx_2d_rho_u, idx_2d_rho_w, idx_2d_rho_theta,  \
                        p0, Rd, cpd, cvd, heat_capacity_ratio, gravity)

   t_rhs = compute_rhs(Q_tilda, geom, idx_2d_rho, idx_2d_rho_u, idx_2d_rho_w, idx_2d_rho_theta,  \
                        p0, Rd, cpd, cvd, heat_capacity_ratio, gravity)


   # return rhs
   return rhs - t_rhs

      