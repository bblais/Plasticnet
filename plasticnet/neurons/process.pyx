# post process should go from z to z, not y to z, so that we can 
# make a series of them

from plasticnet.plasticnet cimport *
cimport cython
import pylab

import numpy as np
cimport numpy as np

cdef class add_noise_normal(post_process_neuron):
    cdef public double mean,std

    def __init__(self,double mean=0.0,std=1.0):
        self.mean=mean
        self.std=std
        post_process_neuron.__init__(self)
        self._reset()

    @cython.cdivision(True)
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    cpdef update(self,double t,simulation sim):
        cdef int i
        cdef double *z=<double *>self.n.output.data
        
        for i in range(self.n.N):
            z[i]=z[i]+self.std*randn()+self.mean


cdef class add_noise_multiplicative(post_process_neuron):
    cdef public double std

    def __init__(self,double std=-1,double tau=-1):
        if std<0 and tau<0:
            self.std=1.0
        elif std>=0 and tau<0:
            self.std=std
        elif std<0 and tau>0:
            self.std=sqrt(1.0/2.0/tau)
        else:
            raise ValueError,"std and tau both undefined"

        post_process_neuron.__init__(self)
        self._reset()

    @cython.cdivision(True)
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    cpdef update(self,double t,simulation sim):
        cdef int i
        cdef double *z=<double *>self.n.output.data
        
        for i in range(self.n.N):
            z[i]=z[i]+self.std*randn()*sqrt(abs(z[i]))


cdef class add_noise_uniform(post_process_neuron):
    cdef public double mean,std

    def __init__(self,double mean=0.0,std=1.0):
        self.mean=mean
        self.std=std
        post_process_neuron.__init__(self)
        self._reset()

    @cython.cdivision(True)
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    cpdef update(self,double t,simulation sim):
        cdef int i
        cdef double *z=<double *>self.n.output.data
        
        for i in range(self.n.N):
            z[i]=z[i]+(randu()-0.5)*2.0*1.732050807569*self.std+self.mean

cdef class relative_to_spontaneous(post_process_neuron):
    cdef public double y0

    def __init__(self,double y0):
        self.y0=y0
        post_process_neuron.__init__(self)
        self._reset()

    @cython.cdivision(True)
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    cpdef update(self,double t,simulation sim):
        cdef int i
        cdef double *z=<double *>self.n.output.data
        
        for i in range(self.n.N):
            if z[i]<0.0:
                z[i]=0.0
            z[i]=z[i]-self.y0


## Again, this should go from z to z not y to z
## if it occurs after the non-linearity, then it might be weird

cdef class subtract_beta(post_process_neuron):
    cdef public double tau,a
    cdef public np.ndarray beta,original_output
    
    cpdef _reset(self):
        self.beta=np.zeros(self.n.N,dtype=np.float)    
        self.original_output=np.zeros(self.n.N,dtype=np.float)    
    
    def __init__(self,double tau=100.0,double a=1.0):
        self.tau=tau
        self.a=a
        post_process_neuron.__init__(self)
        self._reset()
        
    @cython.cdivision(True)
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    cpdef update(self,double t,simulation sim):
        cdef int i
        cdef double *y=<double *>self.n.linear_output.data
        cdef double *z=<double *>self.n.output.data
        cdef double *beta=<double *>self.beta.data
        cdef double *zo=<double *>self.original_output.data
        
        for i in range(self.n.N):
            zo[i]=z[i]
            beta[i]+=(z[i]-beta[i])/self.tau
            z[i]-=self.a*beta[i]


cdef class min_max(post_process_neuron):
    cdef public double bottom,top
    def __init__(self,double bottom,double top):
        self.bottom=bottom
        self.top=top
        post_process_neuron.__init__(self)
        
    @cython.cdivision(True)
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    cpdef update(self,double t,simulation sim):
        cdef int i
        cdef double *y=<double *>self.n.linear_output.data
        cdef double *z=<double *>self.n.output.data
        
        for i in range(self.n.N):
            if z[i]<self.bottom:
                z[i]=self.bottom
            elif z[i]>self.top:
                z[i]=self.top
            else:
                pass

cdef class scale_shift(post_process_neuron):
    cdef public double scale,shift
    def __init__(self,double scale,double shift):
        self.scale=scale
        self.shift=shift
        post_process_neuron.__init__(self)
        
    @cython.cdivision(True)
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    cpdef update(self,double t,simulation sim):
        cdef int i
        cdef double *y=<double *>self.n.linear_output.data
        cdef double *z=<double *>self.n.output.data
        
        for i in range(self.n.N):
            z[i]=z[i]*self.scale+self.shift


cdef class sigmoid(post_process_neuron):
    cdef public double bottom,top
    def __init__(self,double bottom,double top):
        self.bottom=bottom
        self.top=top
        post_process_neuron.__init__(self)
        
    @cython.cdivision(True)
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    cpdef update(self,double t,simulation sim):
        cdef int i
        cdef double *y=<double *>self.n.linear_output.data
        cdef double *z=<double *>self.n.output.data
        
        for i in range(self.n.N):
            if z[i]<0:
                z[i]=self.bottom*(2.0/(1.0+exp(-2.0*(z[i]/self.bottom)))-1.0)
            else:
                z[i]=self.top*(2.0/(1.0+exp(-2.0*(z[i]/self.top)))-1.0)
        
        
   