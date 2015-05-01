from plasticnet.plasticnet cimport *
cimport cython
import pylab

import numpy as np
cimport numpy as np

cdef truncate(double x):
    if x>0.0:
        return x
    else:
        return 0

cdef class Miller2014_Eq3(connection):

    cdef public double tau_w,gamma,tau_y,theta
    cdef public double w_max,w_min,yo
    cdef public np.ndarray y_bar
    cdef public np.ndarray initial_y_bar
    cdef public object initial_y_bar_range

    cpdef _reset(self):
        if self.reset_to_initial:
            self.y_bar=self.initial_y_bar.copy()
        else:
            self.y_bar=pylab.rand(self.post.N)*(self.initial_y_bar_range[1]-
                                       self.initial_y_bar_range[0])+self.initial_y_bar_range[0]

        self.initial_y_bar=self.y_bar.copy()
        connection._reset(self)

    def __init__(self,neuron pre,neuron post,initial_weight_range=None,initial_y_bar_range=None):
        if initial_y_bar_range is None:
            self.initial_y_bar_range=[0,.1]
        else:
            self.initial_y_bar_range=initial_y_bar_range

        connection.__init__(self,pre,post,initial_weight_range)

        self.tau_w=0.3  # days - not sure what days is
        self.tau_y=100.0
        self.yo=0.8
        self.gamma=0.23
        self.theta=0.6
        self.w_max=1.0
        self.w_min=0.6
        self.name='Miller2014_Eq3'
        self._reset()


    @cython.cdivision(True)
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    cpdef update(self,double t,simulation sim):
        cdef int __i,__j
    
        cdef double *W=<double *>self.weights.data
        cdef double *y_bar=<double *>self.y_bar.data
        cdef double *X,*Y   # outputs for pre and post

        cdef int __wi
        X=<double *>self.pre.output.data
        Y=<double *>self.post.output.data


        for __i in range(self.post.N):
            for __j in range(self.pre.N):
                __wi=__i*self.pre.N+__j
                W[__wi]+=sim.dt*(truncate(self.w_max-W[__wi])*truncate(X[__j]*Y[__i]-self.theta) -
                                 truncate(W[__wi]-self.w_min)*truncate(self.theta-X[__j]*Y[__i])+
                                 self.gamma*W[__wi]*(1-y_bar[__i]/self.yo))/self.tau_w

        for __i in range(self.post.N):
            y_bar[__i]+=sim.dt*(Y[__i]-y_bar[__i])/self.tau_y




cdef class Miller2014_Eq5(connection):

    cdef public double tau_rho,tau_H,theta
    cdef public double rho_max,rho_min,yo
    cdef public np.ndarray rho,H
    cdef public np.ndarray initial_rho,initial_H
    cdef public object initial_rho_range,initial_H_range

    cpdef _reset(self):
        if self.reset_to_initial:
            self.H=self.initial_H.copy()
            self.rho=self.initial_rho.copy()
        else:
            self.H=pylab.rand(self.post.N)*(self.initial_H_range[1]-
                                       self.initial_H_range[0])+self.initial_H_range[0]
            self.rho=pylab.rand(self.post.N,self.pre.N)*(self.initial_rho_range[1]-
                                       self.initial_rho_range[0])+self.initial_rho_range[0]

        self.initial_H=self.H.copy()
        self.initial_rho=self.rho.copy()
        connection._reset(self)

    def __init__(self,neuron pre,neuron post,initial_rho_range=None,initial_H_range=None):
        if initial_H_range is None:
            self.initial_H_range=[0,.1]
        else:
            self.initial_H_range=initial_H_range

        if initial_rho_range is None:
            self.initial_rho_range=[0,.1]
        else:
            self.initial_rho_range=initial_rho_range


        connection.__init__(self,pre,post,None)

        self.tau_rho=0.3  # days - not sure what days is
        self.tau_H=100.0
        self.yo=0.8
        self.theta=0.6
        self.rho_max=1.0
        self.rho_min=0.6
        self.name='Miller2014_Eq5'
        self._reset()


    @cython.cdivision(True)
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    cpdef update(self,double t,simulation sim):
        cdef int __i,__j
    
        cdef double *W=<double *>self.weights.data
        cdef double *rho=<double *>self.rho.data
        cdef double *H=<double *>self.H.data
        cdef double *X,*Y   # outputs for pre and post

        cdef int __wi
        X=<double *>self.pre.output.data
        Y=<double *>self.post.output.data


        for __i in range(self.post.N):
            for __j in range(self.pre.N):
                __wi=__i*self.pre.N+__j

                rho[__wi]+=sim.dt*(
                    truncate(self.rho_max-rho[__wi])*truncate(X[__j]*Y[__i]-self.theta) -
                    truncate(rho[__wi]-self.rho_min)*truncate(self.theta-X[__j]*Y[__i])
                )

                W[__wi]=rho[__wi]*H[__i]

        for __i in range(self.post.N):
            H[__i]+=sim.dt*H[__i]*(1-Y[__i]/self.yo)/self.tau_H




