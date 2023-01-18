from splikes.splikes cimport *
cimport cython
import matplotlib.pyplot as pylab

import numpy as np
cimport numpy as np

cdef class poisson_julijana(neuron):
    cdef public double tau_ex,r_0,g_0
    cdef public np.ndarray u

    cpdef _reset(self):
        self.u=np.zeros(self.N,dtype=np.float)
        neuron._reset(self)
    
    def __init__(self,N):
        neuron.__init__(self,N)
        self.tau_ex=0.011    #in s, membrane time constant (EPSP timescale)
        self.r_0=0.0  # baseline firing rate
        self.g_0=10.0
        
        self._reset()

    @cython.cdivision(True)
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    cpdef update(self,double t,simulation sim):
        cdef int i,j
        cdef connection c
        cdef double x
        cdef neuron pre
        cdef double *u=<double *>self.u.data
        cdef double *rate=<double *>self.rate.data

        # quantities decay exponentially
        x=-sim.dt/self.tau_ex
        for i in range(self.N):
            u[i]*=exp(x)
    
        cdef double *W    #=<double *>c.w.data
        cdef int *spiking   #=<int *>pre.spiking.data
        
        for c in self.connections_pre:
            pre=c.pre
            W=c.W
            spiking=<int *>pre.spiking.data
        
            if pre.is_spike:
                for j in range(pre.N):
                    if spiking[j]:
                        for i in range(self.N):
                            u[i]+=W[i*pre.N+j]/self.tau_ex  
    
        spiking=<int *>self.spiking.data
        self.is_spike=0
        for i in range(self.N):
            rate[i]=self.r_0+self.g_0*u[i]
                
            if randu()<(rate[i]*sim.dt):
                spiking[i]=1
                self.is_spike=1
                self.post_count+=1
                if self.save_spikes_begin<=t<=self.save_spikes_end:
                    self.saved_spikes.append( (t,i) )
            else:
                spiking[i]=0
