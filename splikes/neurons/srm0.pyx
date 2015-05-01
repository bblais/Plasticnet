from splikes.splikes cimport *
cimport cython
import pylab

import numpy as np
cimport numpy as np
from splikes.neurons.isi_distributions cimport *

cdef class srm0(neuron):
    cdef public double tau,a,rate_slope,rate_offset,tau_beta
    cdef public np.ndarray u,v,beta
    cdef public int smoothed

    cpdef _reset(self):
        self.u=np.zeros(self.N,dtype=np.float)
        self.v=np.zeros(self.N,dtype=np.float)
        self.beta=np.zeros(self.N,dtype=np.float)
        neuron._reset(self)
    
    def __init__(self,N):
        neuron.__init__(self,N)
        self.tau=0.1    #in s, membrane time constant (EPSP timescale)
        self.a=1.0  # baseline firing rate
        self.rate_slope=1.0
        self.rate_offset=0.0
        self.smoothed=False
        self.tau_beta=-1
        self.name='SRM0'
        self._reset()

    @cython.cdivision(True)
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    cpdef update(self,double t,simulation sim):
        cdef int i,j
        cdef connection c
        cdef double x
        cdef neuron pre
        cdef double *beta=<double *>self.beta.data
        cdef double *u=<double *>self.u.data
        cdef double *v=<double *>self.v.data
        cdef double *rate=<double *>self.rate.data

        use_beta=self.tau_beta>0


        # quantities decay exponentially
        x=-sim.dt/self.tau
        if self.smoothed:
            for i in range(self.N):
                v[i]*=exp(x)
                u[i]-=(v[i]-u[i])*x  # x is negative, so do -=
        else:
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
                        if self.smoothed:
                            for i in range(self.N):
                                v[i]+=self.a*W[i*pre.N+j]
                        else:
                            for i in range(self.N):
                                u[i]+=self.a*W[i*pre.N+j]
    
        if use_beta:
            for i in range(self.N):
                beta[i]+=sim.dt*(1.0/self.tau_beta)*(u[i]-beta[i])


        spiking=<int *>self.spiking.data
        self.is_spike=0
        for i in range(self.N):
            rate[i]=self.rate_offset+self.rate_slope*(u[i]-beta[i])
                
            if randu()<(rate[i]*sim.dt):
                spiking[i]=1
                self.is_spike=1
                self.post_count+=1
                if self.save_spikes_begin<=t<=self.save_spikes_end:
                    self.saved_spikes.append( (t,i) )
            else:
                spiking[i]=0


cdef class srm0_isi(srm0):
    cdef public distribution ISI
    cdef public int need_to_reset_last_spike_time

    cpdef _reset(self):
        srm0._reset(self)
        self.need_to_reset_last_spike_time=True
    
    def __init__(self,N,distribution ISI):
        srm0.__init__(self,N)
        self.ISI=ISI
        s=str(ISI)
        s=s.split(' ')[0].split('.')[-1]

        self.name='SRM0 ISI %s' % s

        self._reset()

    @cython.cdivision(True)
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    cpdef update(self,double t,simulation sim):
        cdef int i,j
        cdef connection c
        cdef double x,cdf,pdf,_lambda
        cdef neuron pre
        cdef double *u=<double *>self.u.data
        cdef double *v=<double *>self.v.data
        cdef double *rate=<double *>self.rate.data
        cdef double *last_spike_time=<double *>self.last_spike_time.data

        if self.need_to_reset_last_spike_time:
            for i in range(self.N):
                last_spike_time[i]=-sim.dt    # assume a spike one dt ago
            self.need_to_reset_last_spike_time=False

        # quantities decay exponentially
        x=-sim.dt/self.tau
        if self.smoothed:
            for i in range(self.N):
                v[i]*=exp(x)
                u[i]-=(v[i]-u[i])*x  # x is negative, so do -=
        else:
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
                        if self.smoothed:
                            for i in range(self.N):
                                v[i]+=self.a*W[i*pre.N+j]
                        else:
                            for i in range(self.N):
                                u[i]+=self.a*W[i*pre.N+j]
    
        spiking=<int *>self.spiking.data
        self.is_spike=0
        for i in range(self.N):
            rate[i]=self.rate_offset+self.rate_slope*u[i]
            
            if rate[i]==0.0:
                spiking[i]=0
                continue
                
            x=t-last_spike_time[i]
            self.ISI.set_rate(rate[i]) 


            cdf=self.ISI.cdf(x)
            pdf=self.ISI.pdf(x)
            
            if cdf==1.0:  # guarantee a spike - avoid divide by zero
                self.is_spike=1
                spiking[i]=1
            else:
                _lambda=pdf/(1-cdf)
            
                if randu()<_lambda*sim.dt:
                    self.is_spike=1
                    spiking[i]=1
                else:
                    spiking[i]=0

            if spiking[i]:
                last_spike_time[i]=t
                self.is_spike=1
                self.post_count+=1
                if self.save_spikes_begin<=t<=self.save_spikes_end:
                    self.saved_spikes.append( (t,i) )



