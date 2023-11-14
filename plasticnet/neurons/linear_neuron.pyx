from plasticnet.plasticnet cimport *
cimport cython
import numpy as np
cimport numpy as np

cdef threshold(double y,double theta):
    cdef double z
    z=y-theta
    if z<0:
        z=0

    return z


cdef class dynamic_neuron_with_firing_threshold(neuron):
    cdef public double q,tau_f,tau_y
    cdef public np.ndarray theta_f
    cdef public object initial_theta_f_range

    cpdef _reset(self):
        neuron._reset(self)
        self.theta_f=np.random.rand(self.N)*(self.initial_theta_f_range[1]-
                                    self.initial_theta_f_range[0])+self.initial_theta_f_range[0]


    @cython.cdivision(True)
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    cpdef update(self,double t,simulation sim):
        cdef neuron pre
        cdef connection c
    
        cdef double *x
        cdef double *y=<double *>self.linear_output.data
        cdef double *z=<double *>self.output.data
        cdef double *theta_f=<double *>self.theta_f.data
        cdef double *w    #=<double *>c.w.data
        cdef int i,j
        cdef double sum_x
        
        for i in range(self.N):
            y[i]=0.0
            
        sum_x=0.0
        for c in self.connections_pre:
            pre=c.pre
            w=c.w
            x=<double *>pre.output.data
            for j in range(pre.N):
                for i in range(self.N):
                    y[i]+=x[j]*w[i*pre.N+j]

                sum_x+=x[j]

            for i in range(self.N):
                y[i]+=self.q

                    
        for i in range(self.N):
            z[i]+=sim.dt/self.tau_y*(-z[i] + threshold(y[i],theta_f[i]))
            theta_f[i]+=sim.dt/self.tau_f*(sum_x*y[i]-theta_f[i])
    
    
    def __init__(self,*args,initial_theta_f_range=None):
        neuron.__init__(self,*args) # number of neurons
        self.name='Dynamic Neuron with Firing Threshold'


        if initial_theta_f_range is None:
            self.initial_theta_f_range=np.array([0,.1])
        else:
            self.initial_theta_f_range=np.array(initial_theta_f_range)        



        self.q=0
        self.tau_f=100.0
        self.tau_y=100.0

        self.save_attrs.extend(['q','tau_f'])
        self.save_data.extend(['theta_f',])


cdef class linear_neuron(neuron):
    @cython.cdivision(True)
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    cpdef update(self,double t,simulation sim):
        cdef neuron pre
        cdef connection c
    
        cdef double *x
        cdef double *y=<double *>self.linear_output.data
        cdef double *z=<double *>self.output.data
        cdef double *w    #=<double *>c.w.data
        cdef int i,j
        
        for i in range(self.N):
            y[i]=0.0
            
        for c in self.connections_pre:
            pre=c.pre
            w=c.w
            x=<double *>pre.output.data
            for j in range(pre.N):
                for i in range(self.N):
                    y[i]+=x[j]*w[i*pre.N+j]
                    
        for i in range(self.N):
            z[i]=y[i]
    
    
    def __init__(self,*args):
        neuron.__init__(self,*args) # number of neurons
        self.name='Linear Neuron'