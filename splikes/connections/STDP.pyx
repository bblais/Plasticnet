
from splikes.splikes cimport *
cimport cython
import pylab

import numpy as np
cimport numpy as np
    

# 
# name: STDP
# equations:
#     - dM/dt=-M/tau_minus -a_minus*post  : post
#     - dP/dt=-P/tau_plus  +a_plus*pre    : pre
#     - dW/dt=M*pre*g_max + post*P*g_max
# parameters:
#     - g_max=1
#     - a_plus=0.005
#     - a_minus=0.005*1.05
#     - tau_plus=20
#     - tau_minus=20
# 
cdef class STDP(connection):
    cdef public double tau_minus,a_plus,a_minus,g_max,tau_plus
    cdef public np.ndarray P,M
    cpdef _reset(self):
        self.P=np.zeros(self.pre.N,dtype=float)
        self.M=np.zeros(self.post.N,dtype=float)
        connection._reset(self)

    def __init__(self,neuron pre,neuron post,initial_weight_range=None,state=None):
        connection.__init__(self,pre,post,initial_weight_range,state)
    
        self.tau_minus=20
        self.a_plus=0.005
        self.a_minus=0.00525
        self.g_max=1
        self.tau_plus=20
        self._reset()

    @cython.cdivision(True)
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    cpdef update(self,double t,simulation sim):
        cdef int __i,__j
    
        cdef double *P=<double *>self.P.data
        cdef double *M=<double *>self.M.data
        cdef double tau_minus=self.tau_minus
        cdef double a_plus=self.a_plus
        cdef double a_minus=self.a_minus
        cdef double g_max=self.g_max
        cdef double tau_plus=self.tau_plus

        cdef double *W=self.W
        cdef double *post_rate=<double *>self.post.rate.data
        cdef double *pre_rate=<double *>self.pre.rate.data
        cdef int *pre
        cdef int *post   # spikes for pre and post
        cdef int __wi
        
        
        pre=<int *>self.pre.spiking.data
        post=<int *>self.post.spiking.data
    
        for __i in range(self.post.N):
            M[__i]+=sim.dt*(-M[__i]/tau_minus-a_minus*post[__i]/sim.dt)
        for __j in range(self.pre.N):
            P[__j]+=sim.dt*(-P[__j]/tau_plus+a_plus*pre[__j]/sim.dt)
        for __i in range(self.post.N):
            for __j in range(self.pre.N):
                __wi=__i*self.pre.N+__j
                W[__wi]+=sim.dt*(M[__i]*pre[__j]/sim.dt*g_max+post[__i]/sim.dt*P[__j]*g_max)
        
        self.apply_weight_limits()
        