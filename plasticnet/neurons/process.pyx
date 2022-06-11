# post process should go from z to z, not y to z, so that we can 
# make a series of them

from plasticnet.plasticnet cimport *
cimport cython
import pylab

import numpy as np
cimport numpy as np


cdef class temporal_filter(post_process_neuron):

    cdef public list buffer
    cdef public np.ndarray filter
    cdef public np.ndarray B

    cpdef _reset(self):
        self.buffer=[]

    def __init__(self,filter):
        self.filter=filter
        
        post_process_neuron.__init__(self)
        self._reset()

        self.save_attrs+=['filter']

    @cython.cdivision(True)
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    cpdef update(self,double t,simulation sim):
        cdef int i,k,L,Lb,N,k2
        cdef double *z=<double *>self.n.output.data
        cdef double *zb
        cdef double F
        
        self.buffer.append(self.n.output.copy())
        Lb=len(self.buffer)
        L=len(self.filter)

        if Lb>L:
            self.buffer.pop(0)
        Lb=len(self.buffer)

        assert (Lb<=L),f"buffer {Lb} and filter {L}"

        self.B=np.array(self.buffer)  # inefficient but I'll go with it
        zb=<double *>self.B.data  # most recent at the end

        N=self.n.N
        if self.verbose:
            print("Update")

        for k in range(Lb):
            F=self.filter[k]  # most recent at the start
            k2=Lb-1-k
        
            if self.verbose:
                print(f"k={k} k2={k2} F={F}")

            if k==0:
                for i in range(N):
                    z[i]=zb[i+k2*N]*F

                    if self.verbose:
                        if i<5:
                            print(f"\t {i}: {zb[i+k2*N]} x {F} = {zb[i+k2*N]*F} ")

                if self.verbose:
                    print("\t...")

            else:
                for i in range(N):
                    z[i]+=zb[i+k2*N]*F

                    if self.verbose:
                        if i<5:
                            print(f"\t {i}: {zb[i+k2*N]} x {F} = {zb[i+k2*N]*F} ")

                if self.verbose:
                    print("\t...")

            if self.verbose:
                print()
                for i in range(5):
                    print(f"\t z[{i}]={z[i+k2*N]} ")

                print("...")

cdef class zero_fraction(post_process_neuron):

    cdef public fraction
    def __init__(self,double fraction=0.0):
        self.fraction=fraction
        post_process_neuron.__init__(self)
        self._reset()

        self.save_attrs+=['fraction']

    @cython.cdivision(True)
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    cpdef update(self,double t,simulation sim):
        cdef int i
        cdef double *z=<double *>self.n.output.data
        
        for i in range(self.n.N):
            if randu()<=self.fraction:
                z[i]=0.0


cdef class add_noise_normal(post_process_neuron):
    cdef public double mean,std

    def __init__(self,double mean=0.0,std=1.0):
        self.mean=mean
        self.std=std
        post_process_neuron.__init__(self)
        self._reset()

        self.save_attrs+=['mean','std']


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
        self.save_attrs+=['std']

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
        self.save_attrs+=['mean','std']

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
        self.save_attrs+=['y0']

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
        self.beta=np.zeros(self.n.N,dtype=float)    
        self.original_output=np.zeros(self.n.N,dtype=float)    
    
    def __init__(self,double tau=100.0,double a=1.0):
        self.tau=tau
        self.a=a
        post_process_neuron.__init__(self)
        self._reset()
        self.save_attrs+=['tau','a']
        self.save_data+=['beta','original_output']
        
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
            beta[i]+=sim.dt*(z[i]-beta[i])/self.tau
            z[i]-=self.a*beta[i]


cdef class min_max(post_process_neuron):
    cdef public double bottom,top
    def __init__(self,double bottom,double top):
        self.bottom=bottom
        self.top=top
        post_process_neuron.__init__(self)
        self.save_attrs+=['bottom','top']
        
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
        self.save_attrs+=['scale','shift']
        
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
        self.save_attrs+=['bottom','top']
        
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
        
from libc.math cimport log

cdef class log_transform(post_process_neuron):
    cdef public double scale,shift
    def __init__(self,double scale,double shift):
        self.scale=scale
        self.shift=shift
        post_process_neuron.__init__(self)
        self.save_attrs+=['scale','shift']
        
    @cython.cdivision(True)
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    cpdef update(self,double t,simulation sim):
        cdef int i
        cdef double value
        cdef double *y=<double *>self.n.linear_output.data
        cdef double *z=<double *>self.n.output.data
        
        for i in range(self.n.N):
            value=z[i]*self.scale + self.shift
            if value<0.0:
                value=0.0

            z[i]= log(value+1)
        
        
      