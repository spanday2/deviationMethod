import os
import numpy
import matplotlib.pyplot as plt

from common.definitions     import idx_2d_rho       as RHO,           \
                                   idx_2d_rho_w     as RHO_W,         \
                                   idx_2d_rho_u     as RHO_U,         \
                                   idx_2d_rho_theta as RHO_THETA
from common.definitions     import *
from common.graphx          import image_field
from common.program_options import Configuration
from geometry               import Geometry

def output_step(Q: numpy.ndarray, geom: Geometry, param: Configuration, filename: str, step_id) -> None:
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
      
      image_field(geom, Theta, filename, 303.1, 303.7, 7)

   elif param.case_number == 666:
      
      Q_base = numpy.zeros_like(Q)
      # T0      = 300.0                                      # temperature
      # H       = Rd * T0 / gravity                          # scale height
      # t = T0
      # pressure = p0 * numpy.exp(-geom.X3 / H)
      # Q_base[idx_2d_rho] = pressure / (Rd * t)
      # Q_base[idx_2d_rho_theta] = Q_base[idx_2d_rho] * t * (p0 / pressure)**(Rd/cpd)  

      Q_total = Q + Q_base

      
      # Calculate the total Q vector
      w                         = Q_total[RHO_W,:,:] / Q_total[RHO,:,:]
      u                         = Q_total[RHO_U,:,:] / Q_total[RHO,:,:]
      rho                       = Q_total[RHO]

      pressure = p0 * numpy.exp((cpd/cvd) * numpy.log((Rd/p0)*Q_total[idx_2d_rho_theta, :, :]))
      
      c = numpy.sqrt(heat_capacity_ratio*pressure / rho)
      M = (numpy.sqrt(u**2+w**2) / c).max()
      print("{:.5e}".format(M))
      array = numpy.array([f"{M:.16e}"])   
      #Open the file in append mode and write the new values
      with open("config25m_2.txt", "a") as file:
         # Convert array to string and append to the file
         file.write(" ".join(map(str, array)) + "\n")
      # if step_id > 0:
      #    image_field(geom, w, filename, numpy.min(w), numpy.max(w), 20)

      # plt.figure(figsize=(6, 4))
      # plt.plot(Q_total[2,:,20] / Q_total[0,:,20], geom.X3, 'bo-')
      # # plt.plot(Q[0,:,20], geom.X3, 'b-')
      # plt.xlabel('w')
      # plt.ylabel('z')
      # plt.grid(True)
      # # Save the figure
      # plt.savefig(filename)
      # plt.close() 


   elif param.case_number == 3:
      image_field(geom, (Q[RHO_THETA,:,:] / Q[RHO,:,:]), filename, 303., 303.7, 8)
   elif param.case_number == 4:
      image_field(geom, (Q[RHO_THETA,:,:] / Q[RHO,:,:]), filename, 290., 300., 10)
