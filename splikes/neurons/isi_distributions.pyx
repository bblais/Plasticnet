from splikes.splikes cimport *
cimport cython
import matplotlib.pyplot as pylab

import numpy as np
cimport numpy as np

        
cdef class distribution:
    cpdef pdf(self,double x):
        return 0
    cpdef cdf(self,double x):
        return 0
    cpdef set_rate(self,double rate):
        pass

cdef class invgauss(distribution):
    
    def __init__(self,mu,_lambda):
        self._lambda=_lambda
        self.mu=mu
        self.pi=np.pi
    cpdef pdf(self,double x):
        cdef double pi=self.pi
        return sqrt(self._lambda/(2*pi*x**3))*exp(-self._lambda*(x-self.mu)**2/(2*self.mu**2*x))


    cpdef cdf(self,double x):
        return phi(sqrt(self._lambda/x)*(x/self.mu-1))+exp(2*self._lambda/self.mu)*phi(-sqrt(self._lambda/x)*(x/self.mu+1))
    
    cpdef set_rate(self,double rate):
        if rate>0.0:
            self.mu=1.0/rate
        else:
            self.mu=1e300
    
cdef class exponential(distribution):
    
    def __init__(self,_lambda):
        self._lambda=_lambda

    cpdef pdf(self,double x):
        return self._lambda*exp(-self._lambda*x)

    cpdef cdf(self,double x):
        return 1-exp(-self._lambda*x)
    
    cpdef set_rate(self,double rate):
        self._lambda=rate

    
    
cdef class normal(distribution):
    
    def __init__(self,double sigma):
        self.sigma=sigma
        self.mu=0.0

    cpdef pdf(self,double x):
        cdef double pi=3.14159265358979323846264338327950288

        return 1/sqrt(2*pi)/self.sigma*exp(-(x-self.mu)**2/2/self.sigma/self.sigma)

    cpdef cdf(self,double x):
        return 0.5*(1+erf((x-self.mu)/sqrt(2)/self.sigma))
    
    cpdef set_rate(self,double rate):
        self.mu=1.0/rate

    
cdef class uniform(distribution):
    def __init__(self,double a,double b):
        self.a=a
        self.b=b
        self.mean=0.0
        assert a<0
        assert b>0

    cpdef pdf(self,double x):
        if x<self.mean+self.a:
            return 0.0
        if x>self.mean+self.b:
            return 0.0

        return 1.0/(self.b-self.a)

    cpdef cdf(self,double x):
        if x<self.mean+self.a:
            return 0.0
        if x>self.mean+self.b:
            return 1.0

        return (x-self.a-self.mean)/(self.b-self.a)
    
    cpdef set_rate(self,double rate):
        self.mean=1.0/rate

    
