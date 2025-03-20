import os
import numpy

from common.definitions     import idx_2d_rho       as RHO,           \
                                   idx_2d_rho_w     as RHO_W,         \
                                   idx_2d_rho_u     as RHO_U,         \
                                   idx_2d_rho_theta as RHO_THETA
from common.definitions     import gravity, Rd, cvd, cpd, heat_capacity_ratio, p0
from common.graphx          import image_field
from common.program_options import Configuration
from geometry               import Geometry

def output_step(Q: numpy.ndarray, geom: Geometry, param: Configuration, filename: str) -> None:
   if param.case_number == 0:
      image_field(geom, (Q[RHO_W,:,:]), filename, -1, 1, 25, label='w (m/s)', colormap='bwr')

   elif param.case_number <= 2:
      # Calculate the base state
      theta_base                = numpy.ones_like(geom.X1)*param.bubble_theta
      exner_base                = (1.0 - gravity / (cpd * theta_base) * geom.X3)
      rho_base                  = p0 / (Rd * theta_base) * exner_base**(cvd / Rd)
      E_base                    = cvd*theta_base*exner_base + gravity*geom.X3    # We did not add 0.5*(u^2+w^2) because its zero
      Q_tilda                   = numpy.zeros_like(Q)
      Q_tilda[RHO]              = rho_base
      Q_tilda[RHO_THETA]        = rho_base * E_base

      # Calculate the total Q vector
      Q_total                   = Q + Q_tilda

      # Convert Energy to potential temperature
      e                         = Q_total[RHO_THETA,:,:] / Q_total[RHO,:,:]
      w                         = Q_total[RHO_W,:,:] / Q_total[RHO,:,:]
      u                         = Q_total[RHO_U,:,:] / Q_total[RHO,:,:]
      rho                       = Q_total[RHO]
      pressure                  = (heat_capacity_ratio-1)*(Q_total[RHO_THETA] - 0.5*rho*(u**2+w**2) - rho*gravity*geom.X3)
      exner                     = (pressure/p0)**(Rd/cpd)
      Theta                     =  1/(cvd*exner)*(e - 0.5*(u**2 + w**2) - gravity*geom.X3)

      # Plot potential temperature
      image_field(geom, Theta, filename, 303.1, 303.7, 7)

   elif param.case_number == 3:
      image_field(geom, (Q[RHO_THETA,:,:] / Q[RHO,:,:]), filename, 303., 303.7, 8)
   elif param.case_number == 4:
      image_field(geom, (Q[RHO_THETA,:,:] / Q[RHO,:,:]), filename, 290., 300., 10)
