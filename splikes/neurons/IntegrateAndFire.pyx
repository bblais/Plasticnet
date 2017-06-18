from splikes.splikes cimport *
cimport cython
import pylab

import numpy as np
cimport numpy as np


cdef class IntegrateAndFire(neuron):
    cdef public double reset,tau_m,tau_in,V_rev_inh,V_rev_exc,tau_ex,V_rest,threshold
    cdef public np.ndarray g_e,g_i,V
    cpdef _reset(self):
        self.g_e=np.zeros(self.N,dtype=np.float)
        self.g_i=np.zeros(self.N,dtype=np.float)
        self.V=np.zeros(self.N,dtype=np.float)
        neuron._reset(self)

    def __init__(self,N):
        neuron.__init__(self,N)
    
        self.reset=-65.0
        self.tau_m=20.0
        self.tau_in=5.0
        self.V_rev_inh=-65.0
        self.V_rev_exc=0.0
        self.tau_ex=5.0
        self.V_rest=-65.0
        self.threshold=-55.0
        self.name='Integrate and Fire'
        
        self._reset()

    @cython.cdivision(True)
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    cpdef update(self,double t,simulation sim):
        cdef int __i,__j
        cdef connection c
        cdef neuron pre
    
        cdef double *g_e=<double *>self.g_e.data
        cdef double *g_i=<double *>self.g_i.data
        cdef double *V=<double *>self.V.data
        cdef double reset=self.reset
        cdef double tau_m=self.tau_m
        cdef double tau_in=self.tau_in
        cdef double V_rev_inh=self.V_rev_inh
        cdef double V_rev_exc=self.V_rev_exc
        cdef double tau_ex=self.tau_ex
        cdef double V_rest=self.V_rest
        cdef double threshold=self.threshold

        cdef double *W,*state
        cdef double spike_scale
        cdef int *spiking   
    
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
    
        for __i in range(self.N):
            V[__i]+=sim.dt*(-(V[__i]-V_rest)/tau_m+g_e[__i]*(V_rev_exc-V[__i])/tau_m+g_i[__i]*(V_rev_inh-V[__i])/tau_m)
            g_e[__i]+=sim.dt*(-g_e[__i]/tau_ex)
            g_i[__i]+=sim.dt*(-g_i[__i]/tau_in)
        
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
            else:
                spiking[__i]=0            
    


cdef class IntegrateAndFire_Gavornik2009(neuron):

    # C dv_i/dt = gl (El-v_i) + g_E_i (E_e-v_i)

    # synaptic activation of kth pre-syn neuron
    # dsk/dt = - sk/tau_s + rho (1-sk)+delta(spike_k)
    # g_Ei = sum_k L_ik sk

    cdef public double reset,tau_m,tau_in,
    cdef public double V_rev_inh,V_rev_exc,tau_ex,V_rest,threshold
    cdef public np.ndarray g_e,g_i,V
    cpdef _reset(self):
        self.g_e=np.zeros(self.N,dtype=np.float)
        self.g_i=np.zeros(self.N,dtype=np.float)
        self.V=np.zeros(self.N,dtype=np.float)
        neuron._reset(self)

    def __init__(self,N):
        neuron.__init__(self,N)
    
        self.reset=-65.0
        self.tau_m=20.0
        self.tau_in=5.0
        self.V_rev_inh=-65.0
        self.V_rev_exc=0.0
        self.tau_ex=5.0
        self.V_rest=-65.0
        self.threshold=-55.0
        self.name='Integrate and Fire'
        
        self._reset()

    @cython.cdivision(True)
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    cpdef update(self,double t,simulation sim):
        cdef int __i,__j
        cdef connection c
        cdef neuron pre
    
        cdef double *g_e=<double *>self.g_e.data
        cdef double *g_i=<double *>self.g_i.data
        cdef double *V=<double *>self.V.data
        cdef double reset=self.reset
        cdef double tau_m=self.tau_m
        cdef double tau_in=self.tau_in
        cdef double V_rev_inh=self.V_rev_inh
        cdef double V_rev_exc=self.V_rev_exc
        cdef double tau_ex=self.tau_ex
        cdef double V_rest=self.V_rest
        cdef double threshold=self.threshold

        cdef double *W,*state
        cdef double spike_scale
        cdef int *spiking   
    
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
    
        for __i in range(self.N):
            V[__i]+=sim.dt*(-(V[__i]-V_rest)/tau_m+g_e[__i]*(V_rev_exc-V[__i])/tau_m+g_i[__i]*(V_rev_inh-V[__i])/tau_m)
            g_e[__i]+=sim.dt*(-g_e[__i]/tau_ex)
            g_i[__i]+=sim.dt*(-g_i[__i]/tau_in)
        
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
            else:
                spiking[__i]=0            
