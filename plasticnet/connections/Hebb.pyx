from plasticnet.plasticnet cimport *
cimport cython
import pylab

import numpy as np
cimport numpy as np

cdef class Hebb(connection):
    cdef public double eta,tau
    cdef public np.ndarray theta
    cdef public np.ndarray initial_theta
    cdef public object initial_theta_range
    
    cpdef _reset(self):
        self.theta=pylab.rand(self.post.N)*(self.initial_theta_range[1]-
                                   self.initial_theta_range[0])+self.initial_theta_range[0]
        self.initial_theta=self.theta.copy()
        connection._reset(self)
    
    
    def __init__(self,neuron pre,neuron post,initial_weight_range=None,initial_theta_range=None):
        if initial_theta_range is None:
            self.initial_theta_range=[0,.1]

        connection.__init__(self,pre,post,initial_weight_range)
        
        self.name='Hebb'
        self.eta=1e-5
        self.tau=100.0
        self._reset()
    
    
    @cython.cdivision(True)
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    cpdef update(self,double t,simulation sim):
        cdef int __i,__j
    
        cdef double *W=<double *>self.weights.data
        cdef double *theta=<double *>self.theta.data
        cdef double *X
        cdef double *Y   # outputs for pre and post
        cdef int __wi
        cdef double eta=self.eta
        cdef double tau=self.tau
        
        X=<double *>self.pre.output.data
        Y=<double *>self.post.output.data
    
        for __i in range(self.post.N):
            for __j in range(self.pre.N):
                __wi=__i*self.pre.N+__j
                W[__wi]+=sim.dt*(eta*X[__j]*Y[__i])
    
        for __i in range(self.post.N):
            theta[__i]+=sim.dt*(Y[__i]*Y[__i]-theta[__i])/tau
