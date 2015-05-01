cdef extern from "math.h":
    double sqrt(double)
    double exp(double)
    double log(double)
    double tanh(double)
    double erf(double)
    double pow(double,double)

cdef class distribution:
    cpdef pdf(self,double x)
    cpdef cdf(self,double x)
    cpdef set_rate(self,double rate)
    
cdef inline phi(x):  # standard normal cdf
    return 0.5*(1+erf(x/sqrt(2)))
    
cdef class invgauss(distribution):
    cdef public double mu,_lambda,pi
    
cdef class exponential(distribution):
    cdef public double _lambda
    
cdef class normal(distribution):
    cdef public double mu,sigma,pi

cdef class uniform(distribution):
    cdef public double a,b,mean
    
