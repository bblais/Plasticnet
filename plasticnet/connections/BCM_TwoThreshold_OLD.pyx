from plasticnet.plasticnet cimport *
cimport cython
import pylab

import numpy as np
cimport numpy as np



cdef class BCM_TwoThreshold_OLD(connection):
    cdef public double eta,tau,tau_L
    cdef public np.ndarray theta,theta_L
    cdef public np.ndarray initial_theta,initial_theta_L
    cdef public object initial_theta_range,initial_theta_L_range
    
    cpdef _reset(self):
        if self.reset_to_initial:
            self.theta=self.initial_theta.copy()
            self.theta_L=self.initial_theta_L.copy()
        else:
            self.theta=pylab.rand(self.post.N)*(self.initial_theta_range[1]-
                                       self.initial_theta_range[0])+self.initial_theta_range[0]
            self.theta_L=pylab.rand(self.post.N)*(self.initial_theta_L_range[1]-
                                       self.initial_theta_L_range[0])+self.initial_theta_L_range[0]

        self.initial_theta=self.theta.copy()
        self.initial_theta_L=self.theta_L.copy()
        connection._reset(self)
    
    
    def __init__(self,neuron pre,neuron post,
        initial_weight_range=None,initial_theta_range=None,
        initial_theta_L_range=None):
        if initial_theta_range is None:
            self.initial_theta_range=[0,.1]
        else:
            self.initial_theta_range=initial_theta_range

        if initial_theta_L_range is None:
            self.initial_theta_L_range=[0,0]
        else:
            self.initial_theta_L_range=initial_theta_L_range

        connection.__init__(self,pre,post,initial_weight_range)
        
        self.name='BCM Two Threshold'
        self.eta=1e-5
        self.tau=100.0
        self.tau_L=-1
        
        self._reset()
    
    
    @cython.cdivision(True)
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    cpdef update(self,double t,simulation sim):
        cdef int __i,__j
    
        cdef double *W=<double *>self.weights.data
        cdef double *theta=<double *>self.theta.data
        cdef double *theta_L=<double *>self.theta_L.data
        cdef double *X
        cdef double *Y   # outputs for pre and post
        cdef double y_offset=0.0,y
        cdef int __wi
        cdef double eta=self.eta
        cdef double tau=self.tau
        cdef double tau_L=self.tau_L
        
        if tau_L<0.0:
            tau_L=tau
        
        X=<double *>self.pre.output.data
        Y=<double *>self.post.output.data
    
        for __i in range(self.post.N):
            for __j in range(self.pre.N):
                __wi=__i*self.pre.N+__j
                if Y[__i]>theta_L[__i]:  # only modify above the lower threshold
                    W[__wi]+=sim.dt*(eta*X[__j]*(Y[__i]-theta_L[__i])*(Y[__i]-theta[__i]))
    
        for __i in range(self.post.N):
            theta[__i]+=sim.dt*(Y[__i]*Y[__i]-theta[__i])/tau
            if y_offset>0.0:
                theta_L[__i]+=sim.dt*(1.0)/tau_L
            else:
                theta_L[__i]+=sim.dt*(-1.0)/tau_L

            # the lower threshold can't go lower than the lower range
            # unless the upper threshold is lower than that
            if theta_L[__i]<self.initial_theta_L_range[0]:
                theta_L[__i]=self.initial_theta_L_range[0]

            # the lower threshold can't be higher than the upper one
            if theta_L[__i]>theta[__i]:
                theta_L[__i]=theta[__i]