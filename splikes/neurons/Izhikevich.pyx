
from splikes.splikes cimport *
cimport cython
import matplotlib.pyplot as pylab

import numpy as np
cimport numpy as np
    
cdef class Izhikevich(neuron):
    cdef public double threshold,reset,d,a,b
    cdef public np.ndarray V,u,total_I
    cpdef _reset(self):
        self.V=np.ones(self.N,dtype=float)*self.reset
        self.u=np.ones(self.N,dtype=float)*self.reset*self.b
        self.total_I=np.zeros(self.N,dtype=float)

        neuron._reset(self)

    def __init__(self,N):
        neuron.__init__(self,N)
    
        self.threshold=30
        self.reset=-65.0
        self.d=8
        self.a=0.02
        self.b=0.2
        self.name='Izhikevich'
        self._reset()
        self.save_attrs.extend(['threshold', 'reset', 'd', 'a', 'b', ])
        self.save_data.extend(['V', 'u', ])

    @cython.cdivision(True)
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    cpdef update(self,double t,simulation sim):
        cdef int __i,__j
        cdef connection c
        cdef neuron pre
    
        cdef double *V=<double *>self.V.data
        cdef double *u=<double *>self.u.data
        cdef double threshold=self.threshold
        cdef double reset=self.reset
        cdef double d=self.d
        cdef double a=self.a
        cdef double b=self.b

        cdef double *W,*state
        cdef double spike_scale
        cdef int *spiking   
    
        cdef double *I
        cdef double *total_I=<double *>self.total_I.data


        for c in self.connections_pre:
            pre=c.pre
            W=c.W
            spiking=<int *>pre.spiking.data
            spike_scale=c.spike_scale
            
            if pre.is_spike and c.use_state:
                state=<double *>c.state.data
                for __j in range(pre.N):
                    if spiking[__j]:
                        for __i in range(self.N):
                            state[__i]+=spike_scale*W[__i*pre.N+__j]    
            if pre.use_I:
                I=<double *>pre.I.data
                for __i in range(self.N):
                    total_I[__i]=0.0
                    for __j in range(pre.N):
                        total_I[__i]+=W[__i*pre.N+__j]*I[__j]
    


    
        for __i in range(self.N):
            V[__i]+=sim.dt*(0.04*V[__i]**2+5*V[__i]+140-u[__i]+total_I[__i])*1000
            u[__i]+=sim.dt*((a*(b*V[__i]-u[__i])))*1000
        
        spiking=<int *>self.spiking.data
        self.is_spike=0
        for __i in range(self.N):
            if V[__i]>self.threshold:
                spiking[__i]=1
                self.is_spike=1
                self.post_count+=1
                if self.save_spikes_begin<=t<=self.save_spikes_end:
                    self.saved_spikes.append( (t,__i) )
                V[__i]=self.reset
                u[__i]+=d
            else:
                spiking[__i]=0            
    