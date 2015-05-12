from plasticnet.plasticnet cimport *
cimport cython
import pylab

import numpy as np
cimport numpy as np


cdef class BCM(connection):
    cdef public double eta,tau,theta_o,gamma,yo
    cdef public np.ndarray theta
    cdef public np.ndarray initial_theta
    cdef public object initial_theta_range
    
    cpdef _reset(self):
        if self.reset_to_initial:
            self.theta=self.initial_theta.copy()
        else:
            self.theta=pylab.rand(self.post.N)*(self.initial_theta_range[1]-
                                       self.initial_theta_range[0])+self.initial_theta_range[0]

        self.initial_theta=self.theta.copy()
        connection._reset(self)
    
    def __init__(self,neuron pre,neuron post,initial_weight_range=None,initial_theta_range=None):
        if initial_theta_range is None:
            self.initial_theta_range=[0,.1]
        else:
            self.initial_theta_range=initial_theta_range
            
        connection.__init__(self,pre,post,initial_weight_range)
        
        self.eta=1e-5
        self.tau=100.0
        self.gamma=0.0
        self.theta_o=1.0
        self.yo=0.0
        self.name='BCM'
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
        cdef double gamma=self.gamma
        cdef double yo=self.yo
        cdef double theta_o=self.theta_o
        
        X=<double *>self.pre.output.data
        Y=<double *>self.post.output.data
    
        for __i in range(self.post.N):
            for __j in range(self.pre.N):
                __wi=__i*self.pre.N+__j
                W[__wi]+=sim.dt*(eta*X[__j]*(Y[__i]-yo)*((Y[__i]-yo)-theta[__i])-eta*gamma*W[__wi])
    
        for __i in range(self.post.N):
            theta[__i]+=sim.dt*(Y[__i]*Y[__i]/theta_o-theta[__i])/tau


cdef class BCM_LawCooper(connection):

    cdef public double eta,tau,theta_o,gamma,yo
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
        else:
            self.initial_theta_range=initial_theta_range

        connection.__init__(self,pre,post,initial_weight_range)
        
        
        self.eta=1e-5
        self.tau=100.0
        self.gamma=0.0
        self.theta_o=1.0
        self.yo=0.0
        self.name='BCM LawCooper'
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
        cdef double gamma=self.gamma        
        cdef double theta_o=self.theta_o
        cdef double yo=self.yo
        cdef double dw
        
        X=<double *>self.pre.output.data
        Y=<double *>self.post.output.data
    
        for __i in range(self.post.N):
            for __j in range(self.pre.N):
                __wi=__i*self.pre.N+__j

                
                
                W[__wi]+=sim.dt*(eta*X[__j]*(Y[__i]-yo)*((Y[__i]-yo)-theta[__i])/theta[__i]-eta*gamma*W[__wi])
    
        for __i in range(self.post.N):
            theta[__i]+=sim.dt*(Y[__i]*Y[__i]/theta_o-theta[__i])/tau


