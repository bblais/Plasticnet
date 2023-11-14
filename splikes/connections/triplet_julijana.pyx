
from splikes.splikes cimport *
cimport cython
import pylab

import numpy as np
cimport numpy as np
    

# 
# name: triplet_julijana
# equations:
#     - dr1/dt=-r1/tau_p + pre  : pre
#     - dr2/dt=-r2/tau_x       : pre
#     - dW/dt=-o1*(A_2m_eff + A_3m*r2)
#     - dr2/dt=pre : pre   # does the order really matter?
#     - dr_of_t/dt = (r_of_t-rate)/tau_r  : post
#     - A_2m_eff = A_2m*r_of_t/pow(rho0,exp_p)  : post
#     - do1/dt=-o1/tau_m + post  : post
#     - dW/dt=r1*(A_2p + A_3p*o2)
#     - do2/dt=-o2/tau_y + post  : post
# parameters:
#     - tau_p = 0.0168       # in s
#     - tau_x = 0.1100       # in s
#     - tau_y = 0.1140       # in s    
#     - A_2p = 0
#     - A_2m = 7.1*0.0000001 #small learning rate
#     - A_3p = 6.5*0.0000001
#     - A_3m = 0.0
#     - rho0 = 10.5     #target postsynaptic firing rate, in Hz
#     - exp_p=2.0  # exponent p for BCM map
#     - tau_r=5.0  # in s            
#     - tau_m = 0.0337       # in s  # should this be in connection?
# 
cdef class triplet_julijana(connection):
    cdef public double A_3p,tau_m,A_2p,tau_r,rho0,exp_p,tau_x,tau_y,A_2m,A_3m,tau_p
    cdef public np.ndarray A_2m_eff,r1,r2,r_of_t,o2,o1
    cpdef _reset(self):
        self.A_2m_eff=np.zeros(self.post.N,dtype=np.float)
        self.r1=np.zeros(self.pre.N,dtype=np.float)
        self.r2=np.zeros(self.pre.N,dtype=np.float)
        self.r_of_t=np.zeros(self.post.N,dtype=np.float)
        self.o2=np.zeros(self.post.N,dtype=np.float)
        self.o1=np.zeros(self.post.N,dtype=np.float)
        connection._reset(self)

    def __init__(self,neuron pre,neuron post,initial_weight_range=None,state=None):
        connection.__init__(self,pre,post,initial_weight_range,state)
    
        self.A_3p=6.5e-07
        self.tau_m=0.0337
        self.A_2p=0
        self.tau_r=5.0
        self.rho0=10.5
        self.exp_p=2.0
        self.tau_x=0.11
        self.tau_y=0.114
        self.A_2m=7.1e-07
        self.A_3m=0.0
        self.tau_p=0.0168
        self._reset()

    @cython.cdivision(True)
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    cpdef update(self,double t,simulation sim):
        cdef int __i,__j
    
        cdef double *A_2m_eff=<double *>self.A_2m_eff.data
        cdef double *r1=<double *>self.r1.data
        cdef double *r2=<double *>self.r2.data
        cdef double *r_of_t=<double *>self.r_of_t.data
        cdef double *W=self.W
        cdef double *post_rate=<double *>self.post.rate.data
        cdef double *pre_rate=<double *>self.pre.rate.data
        cdef double *o2=<double *>self.o2.data
        cdef double *o1=<double *>self.o1.data
        cdef double A_3p=self.A_3p
        cdef double tau_m=self.tau_m
        cdef double A_2p=self.A_2p
        cdef double tau_r=self.tau_r
        cdef double rho0=self.rho0
        cdef double exp_p=self.exp_p
        cdef double tau_x=self.tau_x
        cdef double tau_y=self.tau_y
        cdef double A_2m=self.A_2m
        cdef double A_3m=self.A_3m
        cdef double tau_p=self.tau_p

        cdef int *pre
        cdef int *post   # spikes for pre and post
        cdef int __wi
        
        
        pre=<int *>self.pre.spiking.data
        post=<int *>self.post.spiking.data
    
        cdef double x
        x=-sim.dt/tau_m
        for __i in range(self.post.N):
            o1[__i]*=exp(x)
        
        x=-sim.dt/tau_y
        for __i in range(self.post.N):
            o2[__i]*=exp(x)

    
        for __j in range(self.pre.N):
            r1[__j]+=sim.dt*(-r1[__j]/tau_p)
            r1[__j]+=pre[__j]
            r2[__j]+=sim.dt*(-r2[__j]/tau_x)
            
        for __i in range(self.post.N):
            for __j in range(self.pre.N):
                __wi=__i*self.pre.N+__j
                W[__wi]+=sim.dt*(-o1[__i]*(A_2m_eff[__i]+A_3m*r2[__j]))
                
        self.apply_weight_limits()
                
        for __j in range(self.pre.N):
            r2[__j]+=pre[__j]
            
        for __i in range(self.post.N):
            r_of_t[__i] = r_of_t[__i] + ( pow(post_rate[__i],exp_p) -  r_of_t[__i])*sim.dt/self.tau_r
            A_2m_eff[__i]=(A_2m*r_of_t[__i]/pow(rho0,exp_p))
            o1[__i]+=post[__i]
        for __i in range(self.post.N):
            for __j in range(self.pre.N):
                __wi=__i*self.pre.N+__j
                W[__wi]+=sim.dt*(r1[__j]*(A_2p+A_3p*o2[__i]))
        for __i in range(self.post.N):
            o2[__i]+=post[__i]
        
        self.apply_weight_limits()
        