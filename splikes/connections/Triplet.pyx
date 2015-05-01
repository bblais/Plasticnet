
from splikes.splikes cimport *
cimport cython
import pylab

import numpy as np
cimport numpy as np
    

# 
# name: Gerstner2006
# equations:
#     - dx_fast/dt=-x_fast/tau_x_fast + pre         : pre
#     - dy_fast/dt=-y_fast/tau_y_fast +post     : post
#     - dW/dt=-pre*y_fast*(A2_minus + A3_minus*x_slow)+post*x_fast*(A2_plus+A3_plus*y_slow)
#     - dx_slow/dt=-x_slow/tau_x_slow + pre         : pre
#     - dy_slow/dt=-y_slow/tau_y_slow +post     : post
# parameters:
#     - A2_plus=0
#     - tau_y_slow=114.0
#     - A3_minus=0
#     - A2_minus=0.0071
#     - tau_y_fast=33.7
#     - tau_x_fast=16.8
#     - A3_plus=0.0065
#     - tau_x_slow=946.0
# 
cdef class Gerstner2006(connection):
    cdef public double tau_x_slow,tau_y_slow,A3_minus,A2_minus,tau_y_fast,tau_x_fast,A3_plus,A2_plus
    cdef public double eta
    cdef public np.ndarray y_fast,x_fast,y_slow,x_slow
    cpdef _reset(self):
        self.y_fast=np.zeros(self.post.N,dtype=np.float)
        self.x_fast=np.zeros(self.pre.N,dtype=np.float)
        self.y_slow=np.zeros(self.post.N,dtype=np.float)
        self.x_slow=np.zeros(self.pre.N,dtype=np.float)
        connection._reset(self)

    def __init__(self,neuron pre,neuron post,initial_weight_range=None,state=None):
        connection.__init__(self,pre,post,initial_weight_range,state)
        self.name='Gerstner2006'
    
        self.tau_x_slow=0.946
        self.tau_y_slow=0.114
        self.A3_minus=0
        self.eta=1.0
        self.A2_minus=0.0071
        self.tau_y_fast=0.0337
        self.tau_x_fast=0.0168
        self.A3_plus=0.0065
        self.A2_plus=0
        self._reset()

    @cython.cdivision(True)
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    cpdef update(self,double t,simulation sim):
        cdef int __i,__j
    
        cdef double *y_fast=<double *>self.y_fast.data
        cdef double *x_fast=<double *>self.x_fast.data
        cdef double *y_slow=<double *>self.y_slow.data
        cdef double *x_slow=<double *>self.x_slow.data
        cdef double tau_x_slow=self.tau_x_slow
        cdef double tau_y_slow=self.tau_y_slow
        cdef double A3_minus=self.A3_minus
        cdef double A2_minus=self.A2_minus
        cdef double tau_y_fast=self.tau_y_fast
        cdef double tau_x_fast=self.tau_x_fast
        cdef double A3_plus=self.A3_plus
        cdef double A2_plus=self.A2_plus
        cdef double eta=self.eta

        cdef double *W=self.W
        cdef double *post_rate=<double *>self.post.rate.data
        cdef double *pre_rate=<double *>self.pre.rate.data
        cdef int *pre,*post   # spikes for pre and post
        cdef int __wi
        
        
        pre=<int *>self.pre.spiking.data
        post=<int *>self.post.spiking.data
    
        for __j in range(self.pre.N):
            x_fast[__j]+=sim.dt*(-x_fast[__j]/tau_x_fast+pre[__j]/sim.dt)
        for __i in range(self.post.N):
            y_fast[__i]+=sim.dt*(-y_fast[__i]/tau_y_fast+post[__i]/sim.dt)
        for __i in range(self.post.N):
            for __j in range(self.pre.N):
                __wi=__i*self.pre.N+__j
                W[__wi]+=sim.dt*eta*(-pre[__j]/sim.dt*y_fast[__i]*(A2_minus+A3_minus*x_slow[__j])+post[__i]/sim.dt*x_fast[__j]*(A2_plus+A3_plus*y_slow[__i]))
        for __j in range(self.pre.N):
            x_slow[__j]+=sim.dt*(-x_slow[__j]/tau_x_slow+pre[__j]/sim.dt)
        for __i in range(self.post.N):
            y_slow[__i]+=sim.dt*(-y_slow[__i]/tau_y_slow+post[__i]/sim.dt)
        
        self.apply_weight_limits()
        
        
cdef class Triplet_BCM(Gerstner2006):
    cdef public np.ndarray y2,theta
    cdef public double tau_y2,y2_o
    
    cpdef _reset(self):
        self.y2=np.zeros(self.post.N,dtype=np.float)
        self.theta=np.zeros(self.post.N,dtype=np.float)
        Gerstner2006._reset(self)
    
    def __init__(self,neuron pre,neuron post,initial_weight_range=None,state=None):
        Gerstner2006.__init__(self,pre,post,initial_weight_range,state)
        self.tau_y2=5.0 # seconds - not sure what the default should be
        self.y2_o=1.0 
        self.name='Triplet BCM'

    @cython.cdivision(True)
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    cpdef update(self,double t,simulation sim):
        cdef int __i,__j
    
        cdef double *y2=<double *>self.y2.data
        cdef double *theta=<double *>self.theta.data
        cdef double y2_val
        cdef double *y_fast=<double *>self.y_fast.data
        cdef double *x_fast=<double *>self.x_fast.data
        cdef double *y_slow=<double *>self.y_slow.data
        cdef double *x_slow=<double *>self.x_slow.data
        cdef double tau_x_slow=self.tau_x_slow
        cdef double tau_y_slow=self.tau_y_slow
        cdef double A3_minus=self.A3_minus
        cdef double A2_minus=self.A2_minus
        cdef double A2_minus_eff
        cdef double tau_y2=self.tau_y2
        cdef double y2_o=self.y2_o
        cdef double tau_y_fast=self.tau_y_fast
        cdef double tau_x_fast=self.tau_x_fast
        cdef double A3_plus=self.A3_plus
        cdef double A2_plus=self.A2_plus
        cdef double eta=self.eta

        cdef double *W=self.W
        cdef double *post_rate=<double *>self.post.rate.data
        cdef double *pre_rate=<double *>self.pre.rate.data
        cdef int *pre,*post   # spikes for pre and post
        cdef int __wi
        
        
        pre=<int *>self.pre.spiking.data
        post=<int *>self.post.spiking.data
    
        for __j in range(self.pre.N):
            x_fast[__j]+=sim.dt*(-x_fast[__j]/tau_x_fast+pre[__j]/sim.dt)
        for __i in range(self.post.N):
            y_fast[__i]+=sim.dt*(-y_fast[__i]/tau_y_fast+post[__i]/sim.dt)

        for __i in range(self.post.N):
            y2_val=y_fast[__i]*y_fast[__i]   #(y_fast[__i]/tau_y_fast)*(y_fast[__i]/tau_y_fast)
            y2[__i]+=sim.dt*(y2_val-y2[__i])/tau_y2


        for __i in range(self.post.N):
            A2_minus_eff=A2_minus*y2[__i]/y2_o
            
            theta[__i]=A2_minus_eff*tau_y_fast/A3_plus/tau_x_fast/tau_y_slow
            
            for __j in range(self.pre.N):
                __wi=__i*self.pre.N+__j
                W[__wi]+=sim.dt*eta*(-pre[__j]/sim.dt*y_fast[__i]*(A2_minus_eff+A3_minus*x_slow[__j])+post[__i]/sim.dt*x_fast[__j]*(A2_plus+A3_plus*y_slow[__i]))
        for __j in range(self.pre.N):
            x_slow[__j]+=sim.dt*(-x_slow[__j]/tau_x_slow+pre[__j]/sim.dt)
        for __i in range(self.post.N):
            y_slow[__i]+=sim.dt*(-y_slow[__i]/tau_y_slow+post[__i]/sim.dt)
        
        self.apply_weight_limits()
        




cdef class Triplet_BCM_LawCooper(Triplet_BCM):
    def __init__(self,neuron pre,neuron post,initial_weight_range=None,state=None):
        Triplet_BCM.__init__(self,pre,post,initial_weight_range,state)
        self.y2_o=19.2
        self.name='Triplet BCM LawCooper'
    
    
    @cython.cdivision(True)
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    cpdef update(self,double t,simulation sim):
        cdef int __i,__j
    
        cdef double *y2=<double *>self.y2.data
        cdef double *theta=<double *>self.theta.data
        cdef double y2_val
        cdef double *y_fast=<double *>self.y_fast.data
        cdef double *x_fast=<double *>self.x_fast.data
        cdef double *y_slow=<double *>self.y_slow.data
        cdef double *x_slow=<double *>self.x_slow.data
        cdef double tau_x_slow=self.tau_x_slow
        cdef double tau_y_slow=self.tau_y_slow
        cdef double A3_minus=self.A3_minus
        cdef double A2_minus=self.A2_minus
        cdef double A3_plus_eff
        cdef double tau_y2=self.tau_y2
        cdef double y2_o=self.y2_o
        cdef double tau_y_fast=self.tau_y_fast
        cdef double tau_x_fast=self.tau_x_fast
        cdef double A3_plus=self.A3_plus
        cdef double A2_plus=self.A2_plus
        cdef double eta=self.eta

        cdef double *W=self.W
        cdef double *post_rate=<double *>self.post.rate.data
        cdef double *pre_rate=<double *>self.pre.rate.data
        cdef int *pre,*post   # spikes for pre and post
        cdef int __wi
        
        
        pre=<int *>self.pre.spiking.data
        post=<int *>self.post.spiking.data
    
        for __j in range(self.pre.N):
            x_fast[__j]+=sim.dt*(-x_fast[__j]/tau_x_fast+pre[__j]/sim.dt)
        for __i in range(self.post.N):
            y_fast[__i]+=sim.dt*(-y_fast[__i]/tau_y_fast+post[__i]/sim.dt)

        for __i in range(self.post.N):
            y2_val=(y_slow[__i]/tau_y_slow)*(y_slow[__i]/tau_y_slow)
            y2[__i]+=sim.dt*(y2_val-y2[__i])/tau_y2

        for __i in range(self.post.N):
            A3_plus_eff=A3_plus/(y2[__i]/y2_o+1e-4)
            theta[__i]=A2_minus*tau_y_fast/A3_plus_eff/tau_x_fast/tau_y_slow
            
            for __j in range(self.pre.N):
                __wi=__i*self.pre.N+__j
                W[__wi]+=sim.dt*eta*(-pre[__j]/sim.dt*y_fast[__i]*(A2_minus+A3_minus*x_slow[__j])+post[__i]/sim.dt*x_fast[__j]*(A2_plus+A3_plus_eff*y_slow[__i]))
                
        for __j in range(self.pre.N):
            x_slow[__j]+=sim.dt*(-x_slow[__j]/tau_x_slow+pre[__j]/sim.dt)
        for __i in range(self.post.N):
            y_slow[__i]+=sim.dt*(-y_slow[__i]/tau_y_slow+post[__i]/sim.dt)
        
        self.apply_weight_limits()
    