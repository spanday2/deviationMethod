import numpy
import scipy
import math
from time import time
from typing import Callable
import pdb
from common.definitions     import *

from mpi4py import MPI

from common.program_options import Configuration
from .integrator            import Integrator
from solvers                import fgmres, matvec_rat, SolverInfo, newton_krylov
from geometry               import Cartesian2D, DFROperators


class BackwardEuler(Integrator):
   def __init__(self, param: Configuration, rhs_handle: Callable, preconditioner=None) -> None:
      super().__init__(param, preconditioner)
      self.rhs = rhs_handle
      self.tol = param.tolerance
      self.param = param



   def BE_system(self, Q_plus, Q, dt, rhs):
      return (Q_plus - Q) / dt - [rhs(Q_plus)]

   def __step__(self, Q, dt):

      # pdb.set_trace()
      
      #Euler 
      # geom = Cartesian2D((self.param.x0, self.param.x1), (self.param.z0, self.param.z1), self.param.nb_elements_horizontal, self.param.nb_elements_vertical, self.param.nbsolpts,self.param.nb_elements_relief_layer,self.param.relief_layer_height)
      # mtrx = DFROperators(geom, self.param)
      # theta_base = numpy.ones_like(geom.X1)*self.param.bubble_theta
      # exner_base = numpy.zeros_like(theta_base)
      # nk, ni = geom.X1.shape
      # for k in range(nk):
      #   for i in range(ni):
      #       exner_base[k,i] = 1.0 - gravity / (cpd * theta_base[k,i]) * geom.X3[k,i]
      # P_base = p0 * exner_base**(cpd/Rd)
      # t_base = exner_base * theta_base
      # base_state = numpy.zeros((4, nk, ni))
      # base_state[0] = theta_base                     # theta
      # base_state[1] = P_base / (Rd * t_base)         # density P = r R T
      # base_state[2] = P_base                         # pressure
      # base_state[3] = base_state[0]*base_state[1]    # temp-dens

      # Q_tilda = numpy.zeros_like(Q)
      # Q_tilda[0] = base_state[1]
      # Q_tilda[3] = base_state[3]

      #Shallow-water
      # Q_tilda = numpy.zeros_like(Q)
      # Q_tilda[0] = 8000
      
      # deltaQ  = Q - Q_tilda
  

      def BE_fun(Q_plus): return self.BE_system(Q_plus, Q, dt, self.rhs)

      maxiter = None
      if self.preconditioner is not None:
         self.preconditioner.prepare(dt, Q)
         maxiter = 800

      # Update solution
      t0 = time()
      Qnew, nb_iter, residuals = newton_krylov(BE_fun, Q, f_tol=self.tol, fgmres_restart=30,
         fgmres_precond=self.preconditioner, verbose=False, maxiter=maxiter)
      t1 = time()

      self.solver_info = SolverInfo(0, t1 - t0, nb_iter, residuals)
      
      return numpy.reshape(Qnew, Q.shape)
