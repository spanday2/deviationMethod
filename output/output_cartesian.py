import os
import numpy

from common.definitions     import idx_2d_rho       as RHO,           \
                                   idx_2d_rho_w     as RHO_W,         \
                                   idx_2d_rho_theta as RHO_THETA,      \
                                   gravity, cpd, p0, Rd, cvd
from common.graphx          import image_field
from common.program_options import Configuration
from geometry               import Geometry

def output_step(Q: numpy.ndarray, geom: Geometry, param: Configuration, filename: str) -> None:

   if param.case_number == 0:
      image_field(geom, (Q[RHO_W,:,:]), filename, -1, 1, 25, label='w (m/s)', colormap='bwr')
   elif param.case_number < 2:
      image_field(geom, (Q[RHO_THETA,:,:] / Q[RHO,:,:]), filename, 303.1, 303.7, 7)
   elif param.case_number == 2:
      theta_base = numpy.ones_like(geom.X1)*param.bubble_theta
      exner_base = numpy.zeros_like(theta_base)
      nk, ni = geom.X1.shape
      for k in range(nk):
        for i in range(ni):
            exner_base[k,i] = 1.0 - gravity / (cpd * theta_base[k,i]) * geom.X3[k,i]
      P_base = p0 * exner_base**(cpd/Rd)
      t_base = exner_base * theta_base
      Q_tilda = numpy.zeros((4, nk, ni))
      Q_tilda[0] = P_base / (Rd * t_base)         # density P = r R T
      Q_tilda[3] = theta_base*Q_tilda[0]          # temp-dens
      
      Q_total = Q + Q_tilda
      image_field(geom, (Q_total[RHO_THETA,:,:] / Q_total[RHO,:,:]), filename, 303.1, 303.7, 7)

   elif param.case_number == 3:
      image_field(geom, (Q[RHO_THETA,:,:] / Q[RHO,:,:]), filename, 303., 303.7, 8)
   elif param.case_number == 4:
      image_field(geom, (Q[RHO_THETA,:,:] / Q[RHO,:,:]), filename, 290., 300., 10)
