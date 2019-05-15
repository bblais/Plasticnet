from plasticnet.plasticnet cimport *
cimport cython
import pylab

import numpy as np
cimport numpy as np
import sys

from numpy import cos,sin,arctan2,linspace,mgrid,pi,zeros,array,reshape

cdef class tuning_curve(post_process_connection):
    cdef public double k
    cdef public int numang
    cdef object sine_gratings,cosine_gratings
    cdef public object x,y
    cdef public object max_y
    cdef int inputs_per_channel,rf_diameter,num_channels
    cdef public double time_to_next_calc
    cdef public double calc_interval

    def __init__(self,double calc_interval,
                    double k=4.4/13.0*3.141592653589793235,
                    int numang=24):
        self.k=k
        self.numang=numang
        self.calc_interval=calc_interval
        self.time_to_next_calc=0.0
        post_process_connection.__init__(self)

    @cython.cdivision(True)
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    cpdef update(self,double t,simulation sim):
        cdef int count,count_y

        if t<self.time_to_next_calc:
            return 

        self.time_to_next_calc+=self.calc_interval

        pre,post=self.c.pre,self.c.post
        c=self.c

        try:
            rf_diameter=pre.rf_size
        except AttributeError:
            rf_diameter=pre.neuron_list[0].rf_size

        theta=linspace(0.0,pi,self.numang)+pi/2
        x=linspace(0.0,pi,self.numang)*180.0/pi
        
        i,j= mgrid[-rf_diameter//2:rf_diameter//2,
                        -rf_diameter//2:rf_diameter//2]
        i=i+1
        j=j+1
        
        i=i.ravel()
        j=j.ravel()
        
        sine_gratings=[]
        cosine_gratings=[]
        
        for t in theta:
            kx=self.k*cos(t)
            ky=self.k*sin(t)
            
            
            sine_gratings.append(sin(kx*i+ky*j))   # sin grating input (small amp)
            cosine_gratings.append(cos(kx*i+ky*j))   # cos grating input (small amp)


        num_neurons=len(c.weights)

        max_y=[]
        full_y=[]
        for i,w in enumerate(c.weights):  #loop over neurons

            try:
                rf_size=pre.rf_size
                neurons=[pre]
            except AttributeError:
                neurons=pre.neuron_list

            num_channels=len(neurons)

            
            one_max_y=[]
            one_y=[]
            count=0
            for c,ch in enumerate(neurons):   
                rf_size=ch.rf_size
                N=ch.N
                weights_1channel_1neuron=w[count:(count+rf_size*rf_size)]


                y=[]
                for ds,dc in zip(sine_gratings,cosine_gratings):
                
                    cs=(weights_1channel_1neuron*ds).sum() # response to sin/cos grating input
                    cc=(weights_1channel_1neuron*dc).sum()
                    
                    phi=arctan2(cc,cs)  # phase to give max response
                    
                    c=cs*cos(phi)+cc*sin(phi)     # max response
                
                    y.append(c)
                    
                one_max_y.append(max(y))
                one_y.append(y)
          
                count+=rf_size*rf_size

            max_y.append(one_max_y)
            full_y.append(one_y)
                
        self.max_y=array(max_y)

        self.y=array(full_y)


cdef class weight_decay(post_process_connection):
    cdef public double gamma

    def __init__(self,double gamma):
        self.gamma=gamma
        post_process_connection.__init__(self)

    @cython.cdivision(True)
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    cpdef update(self,double t,simulation sim):
        cdef int __i
        cdef int __j
        cdef int __wi
        cdef double *W=<double *>self.c.weights.data
        
        for __i in range(self.c.post.N):
            for __j in range(self.c.pre.N):
                __wi=__i*self.c.pre.N+__j
                W[__wi]-=sim.dt*self.c.eta*self.gamma*W[__wi]



cdef class weight_limits(post_process_connection):
    cdef public double w_max,w_min

    def __init__(self,double w_min,double w_max):
        self.w_min=w_min
        self.w_max=w_max

        post_process_connection.__init__(self)
        
    @cython.cdivision(True)
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    cpdef update(self,double t,simulation sim):
        cdef int __i
        cdef int __j
        cdef int __wi
        cdef double *W=<double *>self.c.weights.data
        
        for __i in range(self.c.post.N):
            for __j in range(self.c.pre.N):
                __wi=__i*self.c.pre.N+__j
                if W[__wi]<self.w_min:
                    W[__wi]=self.w_min

                if W[__wi]>self.w_max:
                    W[__wi]=self.w_max            
   
   
   
cdef class zero_diagonal(post_process_connection):

    cpdef _reset(self):
        if self.c.post.N!=self.c.pre.N:
            raise ValueError,"Zero diagonal can only work for square connections"

    @cython.cdivision(True)
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    cpdef update(self,double t,simulation sim):
        cdef int __i
        cdef int __j
        cdef int __wi
        cdef double *W=<double *>self.c.weights.data

        
        for __i in range(self.c.post.N):
            __wi=__i*self.c.pre.N+__i
            W[__wi]=0.0
   
cdef class normalization(post_process_connection):

    @cython.cdivision(True)
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    cpdef update(self,double t,simulation sim):
        cdef int __i
        cdef int __j
        cdef int __wi
        cdef double sum
        cdef double *W=<double *>self.c.weights.data
        
        for __i in range(self.c.post.N):
            sum=0.0
            for __j in range(self.c.pre.N):
                __wi=__i*self.c.pre.N+__j
                sum+=W[__wi]*W[__wi]
            sum=sqrt(sum)
            for __j in range(self.c.pre.N):
                __wi=__i*self.c.pre.N+__j            
                W[__wi]=W[__wi]/sum

cdef class orthogonalization(post_process_connection):
    cdef public double time_to_next_calc
    cdef public double calc_interval

    def __init__(self,double calc_interval):
        self.calc_interval=calc_interval
        self.time_to_next_calc=0.0

        post_process_connection.__init__(self)


    @cython.cdivision(True)
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    cpdef update(self,double t,simulation sim):
        cdef int __i
        cdef int __j
        cdef int __wi
        cdef int count
        cdef double *W=<double *>self.c.weights.data
        cdef object W2
        cdef object Wp
        cdef object norms

        if t<self.time_to_next_calc:
            return 
        self.time_to_next_calc+=self.calc_interval

        W2=np.matrix(self.c.weights)
        
        for __i in range(1,self.c.post.N): #  "1" is on purpose here: from second neuron on
            Wp=np.array(W2[0:__i,:])
            norms=np.sqrt(np.sum(Wp**2,axis=1))
            norms.shape=(__i,1)
            Wp/=norms  # normalize
            Wp=np.matrix(Wp)
            W2[__i,:]-= W2[__i,:]*Wp.T*Wp

        count=0
        #W2p=<double *>W2.data
        for __i in range(self.c.post.N):
            for __j in range(self.c.pre.N):
                __wi=__i*self.c.pre.N+__j            
                W[__wi]=W2[__i,__j]

