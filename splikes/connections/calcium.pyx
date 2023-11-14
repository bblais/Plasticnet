
from splikes.splikes cimport *
cimport cython
import pylab

import numpy as np
cimport numpy as np

cdef sig(double x,double beta):
    return ((tanh(beta*x/2.0)+1.0)/2.0)
    

# 
# name: calcium
# equations:
#     - dv_backspike_slow/dt=peak_backspike_slow*post-v_backspike_slow/tau_backspike_slow : post
#     - dv_backspike_fast/dt=peak_backspike_fast*post-v_backspike_fast/tau_backspike_fast : post
#     - v_total=Vo+backspike_amplitude*(v_backspike_fast+v_backspike_slow) : post
#     - B=1.0/(1.0 + (exp(mg1 * v_total) / mg2))            : post
#     - h=B*(v_total-v_reversal)                            : post
#     - dI_nmda_slow/dt=i_nmda_mu*(i_nmda_s-I_nmda_slow)*pre - I_nmda_slow/tau_nmda_s : pre
#     - dI_nmda_fast/dt=i_nmda_mu*(i_nmda_f-I_nmda_fast)*pre - I_nmda_fast/tau_nmda_f : pre
#     - dg_nmda/dt=k_plus*g_t-(k_plus+k_minus*(v_total-Vo)**Vp)*g_nmda
#     - I_nmda= g_nmda*(I_nmda_fast + I_nmda_slow) * h    # this is the size of weights
#     - dCa/dt=(I_nmda - Ca/tau_ca)
#     - omega=sig(Ca-alpha2,beta2)-0.5*sig(Ca-alpha1,beta1)
#     - eta=eta_gamma0*Ca
#     - dW/dt=eta*(omega-_lambda*W)
# parameters:
#     - g_nmda_o=-0.0025
#     - g_t=-0.0045
#     - tau_backspike_slow=30
#     - mg2=3.57
#     - mg1=-0.062
#     - i_nmda_f=0.75
#     - eta_gamma0=0.02
#     - v_reversal=130
#     - tau_ca=20
#     - alpha2=0.4
#     - Vp=2
#     - alpha1=0.25
#     - backspike_amplitude=60
#     - tau_backspike_fast=3
#     - i_nmda_s=0.25
#     - i_nmda_mu=0.7
#     - tau_nmda_s=200
#     - peak_backspike_fast=0.75
#     - Vo=-65
#     - peak_backspike_slow=0.25
#     - _lambda=0
#     - tau_nmda_f=50
#     - beta2=20
#     - beta1=60
#     - k_plus=0
#     - k_minus=0
# 
cdef class calcium(connection):
    cdef public double g_t,mg2,mg1,v_reversal,tau_ca,alpha2,alpha1,backspike_amplitude,i_nmda_mu,peak_backspike_fast,peak_backspike_slow,_lambda,beta2,beta1,k_plus,g_nmda_o,tau_backspike_fast,tau_backspike_slow,eta_gamma0,i_nmda_s,tau_nmda_s,Vo,Vp,i_nmda_f,tau_nmda_f,k_minus
    cdef public np.ndarray B,I_nmda,h,Ca,v_total,eta,g_nmda,v_backspike_fast,I_nmda_fast,I_nmda_slow,omega,v_backspike_slow
    cpdef _reset(self):
        self.B=np.zeros(self.post.N,dtype=float)
        self.I_nmda=np.zeros( (self.post.N,self.pre.N),dtype=float)
        self.h=np.zeros(self.post.N,dtype=float)
        self.Ca=np.zeros( (self.post.N,self.pre.N),dtype=float)
        self.v_total=np.zeros(self.post.N,dtype=float)
        self.eta=np.zeros( (self.post.N,self.pre.N),dtype=float)
        self.g_nmda=np.zeros( (self.post.N,self.pre.N),dtype=float)
        self.v_backspike_fast=np.zeros(self.post.N,dtype=float)
        self.I_nmda_fast=np.zeros(self.pre.N,dtype=float)
        self.I_nmda_slow=np.zeros(self.pre.N,dtype=float)
        self.omega=np.zeros( (self.post.N,self.pre.N),dtype=float)
        self.v_backspike_slow=np.zeros(self.post.N,dtype=float)
        connection._reset(self)

    def __init__(self,neuron pre,neuron post,initial_weight_range=None,state=None):
        connection.__init__(self,pre,post,initial_weight_range,state)
    
        self.g_t=-0.0045
        self.mg2=3.57
        self.mg1=-0.062
        self.v_reversal=130
        self.tau_ca=20
        self.alpha2=0.4
        self.alpha1=0.25
        self.backspike_amplitude=60
        self.i_nmda_mu=0.7
        self.peak_backspike_fast=0.75
        self.peak_backspike_slow=0.25
        self._lambda=0
        self.beta2=20
        self.beta1=60
        self.k_plus=0
        self.g_nmda_o=-0.0025
        self.tau_backspike_fast=3
        self.tau_backspike_slow=30
        self.eta_gamma0=0.02
        self.i_nmda_s=0.25
        self.tau_nmda_s=200
        self.Vo=-65
        self.Vp=2
        self.i_nmda_f=0.75
        self.tau_nmda_f=50
        self.k_minus=0
        self._reset()

    @cython.cdivision(True)
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    cpdef update(self,double t,simulation sim):
        cdef int __i,__j
    
        cdef double *B=<double *>self.B.data
        cdef double *I_nmda=<double *>self.I_nmda.data
        cdef double *h=<double *>self.h.data
        cdef double *Ca=<double *>self.Ca.data
        cdef double *v_total=<double *>self.v_total.data
        cdef double *eta=<double *>self.eta.data
        cdef double *g_nmda=<double *>self.g_nmda.data
        cdef double *v_backspike_fast=<double *>self.v_backspike_fast.data
        cdef double *I_nmda_fast=<double *>self.I_nmda_fast.data
        cdef double *I_nmda_slow=<double *>self.I_nmda_slow.data
        cdef double *omega=<double *>self.omega.data
        cdef double *v_backspike_slow=<double *>self.v_backspike_slow.data
        cdef double g_t=self.g_t
        cdef double mg2=self.mg2
        cdef double mg1=self.mg1
        cdef double v_reversal=self.v_reversal
        cdef double tau_ca=self.tau_ca
        cdef double alpha2=self.alpha2
        cdef double alpha1=self.alpha1
        cdef double backspike_amplitude=self.backspike_amplitude
        cdef double i_nmda_mu=self.i_nmda_mu
        cdef double peak_backspike_fast=self.peak_backspike_fast
        cdef double peak_backspike_slow=self.peak_backspike_slow
        cdef double _lambda=self._lambda
        cdef double beta2=self.beta2
        cdef double beta1=self.beta1
        cdef double k_plus=self.k_plus
        cdef double g_nmda_o=self.g_nmda_o
        cdef double tau_backspike_fast=self.tau_backspike_fast
        cdef double tau_backspike_slow=self.tau_backspike_slow
        cdef double eta_gamma0=self.eta_gamma0
        cdef double i_nmda_s=self.i_nmda_s
        cdef double tau_nmda_s=self.tau_nmda_s
        cdef double Vo=self.Vo
        cdef double Vp=self.Vp
        cdef double i_nmda_f=self.i_nmda_f
        cdef double tau_nmda_f=self.tau_nmda_f
        cdef double k_minus=self.k_minus

        cdef double *W=self.W
        cdef double *post_rate=<double *>self.post.rate.data
        cdef double *pre_rate=<double *>self.pre.rate.data
        cdef int *pre
        cdef int *post   # spikes for pre and post
        cdef int __wi
        
        
        pre=<int *>self.pre.spiking.data
        post=<int *>self.post.spiking.data
    
        for __i in range(self.post.N):
            v_backspike_slow[__i]+=sim.dt*(peak_backspike_slow*post[__i]/sim.dt-v_backspike_slow[__i]/tau_backspike_slow)
            v_backspike_fast[__i]+=sim.dt*(peak_backspike_fast*post[__i]/sim.dt-v_backspike_fast[__i]/tau_backspike_fast)
            v_total[__i]=(Vo+backspike_amplitude*(v_backspike_fast[__i]+v_backspike_slow[__i]))
            B[__i]=(1.0/(1.0+(exp(mg1*v_total[__i])/mg2)))
            h[__i]=(B[__i]*(v_total[__i]-v_reversal))
        for __j in range(self.pre.N):
            I_nmda_slow[__j]+=sim.dt*(i_nmda_mu*(i_nmda_s-I_nmda_slow[__j])*pre[__j]/sim.dt-I_nmda_slow[__j]/tau_nmda_s)
            I_nmda_fast[__j]+=sim.dt*(i_nmda_mu*(i_nmda_f-I_nmda_fast[__j])*pre[__j]/sim.dt-I_nmda_fast[__j]/tau_nmda_f)
        for __i in range(self.post.N):
            for __j in range(self.pre.N):
                __wi=__i*self.pre.N+__j
                g_nmda[__wi]+=sim.dt*(k_plus*g_t-(k_plus+k_minus*(v_total[__i]-Vo)**Vp)*g_nmda[__wi])
                I_nmda[__wi]=(g_nmda[__wi]*(I_nmda_fast[__j]+I_nmda_slow[__j])*h[__i])
                Ca[__wi]+=sim.dt*((I_nmda[__wi]-Ca[__wi]/tau_ca))
                omega[__wi]=(sig(Ca[__wi]-alpha2,beta2)-0.5*sig(Ca[__wi]-alpha1,beta1))
                eta[__wi]=(eta_gamma0*Ca[__wi])
                W[__wi]+=sim.dt*(eta[__wi]*(omega[__wi]-_lambda*W[__wi]))
        
        self.apply_weight_limits()
        