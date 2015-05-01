from plasticnet.plasticnet cimport *
cimport cython


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