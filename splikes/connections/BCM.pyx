
from splikes.splikes cimport *
cimport cython
import pylab

import numpy as np
cimport numpy as np

import sys    

# 
# name: BCM_LawCooper
# equations:
#     - dx/dt=-x/tau + a*pre  : pre
#     - dX/dt=(x-X)/tau       : pre
#     - dy/dt=-y/tau + a*post : post
#     - dY/dt=(y-Y)/tau       : post
#     - dtheta/dt=(1.0/tau_thresh)*((Y-yo)*(Y-yo)/theta_o-theta) : post
#     - dW/dt=eta*(X-xo)*((Y-yo)*((Y-yo)-theta)/theta)-gamma*W
# parameters:
#     - a=1
#     - tau=100
#     - xo=0
#     - yo=0
#     - theta_o=2
#     - gamma=0
#     - eta=4e-06
#     - tau_thresh=10000
# 
cdef class BCM_LawCooper(connection):
    cdef public double tau_x,tau_y,xo,yo,theta_o,gamma,eta,
    cdef public tau_theta,tau_beta
    cdef public double ax,ay,scale_x,scale_y
    cdef public bint smoothed_x,smoothed_y
    cdef public np.ndarray theta,beta,y,x,y_avg,X,Y,mod_X,mod_Y,mod_theta,mod_beta
    cdef public object initial_theta_range
    cdef public double time_between_modification,time_to_next_modification
    cdef public int theta_squares_the_average




    cpdef _reset(self):
        self.theta=pylab.rand(self.post.N)*(self.initial_theta_range[1]-
                                   self.initial_theta_range[0])+self.initial_theta_range[0]
        self.beta=pylab.rand(self.post.N)*(self.initial_theta_range[1]-
                                   self.initial_theta_range[0])+self.initial_theta_range[0]

        self.mod_theta=np.zeros(self.post.N,dtype=float)
        self.mod_beta=np.zeros(self.post.N,dtype=float)


        self.y=np.zeros(self.post.N,dtype=float)
        self.Y=np.zeros(self.post.N,dtype=float)
        self.mod_Y=np.zeros(self.post.N,dtype=float)

        self.y_avg=np.sqrt(self.theta)
        self.x=np.zeros(self.pre.N,dtype=float)
        self.X=np.zeros(self.pre.N,dtype=float)
        self.mod_X=np.zeros(self.pre.N,dtype=float)

        self.time_to_next_modification=self.time_between_modification

        connection._reset(self)

    def __init__(self,neuron pre,neuron post,
                        initial_weight_range=None,
                        state=None,
                        initial_theta_range=None,):
                        
                        
        if initial_theta_range is None:
            initial_theta_range=[0.0,0.1]

        self.initial_theta_range=initial_theta_range

        connection.__init__(self,pre,post,initial_weight_range,state)

        self.name='Spiking BCM LawCooper'
        self.tau_x=0.01
        self.tau_y=0.01
        self.scale_x=1.0
        self.scale_y=1.0
        self.ax=1.0/self.tau_x
        self.ay=1.0/self.tau_y
        self.smoothed_x=False
        self.smoothed_y=False

        self.xo=0
        self.yo=0
        self.theta_o=1
        self.gamma=0
        self.eta=4e-06
        self.tau_theta=10
        self.tau_beta=-1
        self.time_between_modification=-1.0
        self.theta_squares_the_average=False
        self._reset()

        self.save_attrs.extend(['tau_x','tau_y','xo','yo','theta_o',
            'gamma','eta','tau_theta','tau_beta','ax','scale_x','scale_y',
            'ay','smoothed_x','smoothed_y','time_between_modification',
            'time_to_next_modification','theta_squares_the_average',])
        self.save_data.extend(['theta','beta','x','y',
            'y_avg','X','Y','mod_X','mod_Y','mod_theta','mod_beta',
            'initial_theta_range'])



    @cython.cdivision(True)
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    cpdef update(self,double t,simulation sim):
        cdef int __i,__j
    
        cdef double *W=self.W
        cdef double *theta=<double *>self.theta.data
        cdef double *beta=<double *>self.beta.data
        cdef double *y=<double *>self.y.data
        cdef double *x=<double *>self.x.data
        cdef double *Y=<double *>self.Y.data
        cdef double *X=<double *>self.X.data
        cdef double *y_avg=<double *>self.y_avg.data

        cdef double *mod_X=<double *>self.mod_X.data
        cdef double *mod_Y=<double *>self.mod_Y.data
        cdef double *mod_theta=<double *>self.mod_theta.data
        cdef double *mod_beta=<double *>self.mod_beta.data

        cdef double tau_x=self.tau_x
        cdef double tau_y=self.tau_y
        cdef double ax=self.ax
        cdef double ay=self.ay
        cdef double scale_x=self.scale_x
        cdef double scale_y=self.scale_y
        cdef double xo=self.xo
        cdef double yo=self.yo
        cdef double theta_o=self.theta_o
        cdef double gamma=self.gamma
        cdef double eta=self.eta
        cdef double tau_theta=self.tau_theta
        cdef double tau_beta=self.tau_beta
        cdef double dt,dw
        cdef double YY,XX
        cdef int modify_now,use_beta

        cdef int *pre
        cdef int *post   # spikes for pre and post
        cdef int __wi
        
        use_beta=self.tau_beta>0

        pre=<int *>self.pre.spiking.data
        post=<int *>self.post.spiking.data
    

        for __j in range(self.pre.N):
            x[__j]+=sim.dt*(-x[__j]/tau_x+ax*pre[__j]/sim.dt)

        if self.smoothed_x:
            for __j in range(self.pre.N):
                X[__j]+=sim.dt*(x[__j]-X[__j])/tau_x
        else:
            for __j in range(self.pre.N):
                X[__j]=x[__j]

        for __i in range(self.post.N):
            y[__i]+=sim.dt*(-y[__i]/tau_y+ay*post[__i]/sim.dt)

        if self.smoothed_y:
            for __i in range(self.post.N):
                Y[__i]+=sim.dt*(y[__i]-Y[__i])/tau_y
        else:
            for __i in range(self.post.N):
                Y[__i]=y[__i]


        for __i in range(self.post.N):
            y_avg[__i]+=sim.dt*((1.0/tau_theta)*((Y[__i]/scale_y/tau_y-yo)-y_avg[__i]))            

        if use_beta:
            for __i in range(self.post.N):
                beta[__i]+=sim.dt*(1.0/tau_beta)*(Y[__i]/scale_y-beta[__i])

        modify_now=False
        if self.time_between_modification<0.0:
            modify_now=True
            dt=sim.dt
        elif t>(self.time_to_next_modification-1e-6):
            modify_now=True
            self.time_to_next_modification+=self.time_between_modification
            dt=self.time_between_modification

        if modify_now:
            for __i in range(self.post.N):
                if use_beta:
                    YY=Y[__i]/scale_y-beta[__i]
                    if YY<0.0:
                        YY=0.0
                else:
                    YY=Y[__i]/scale_y

                mod_Y[__i]=YY
                mod_theta[__i]=theta[__i]
                mod_beta[__i]=beta[__i]


                for __j in range(self.pre.N):
                    XX=X[__j]/scale_x
                    mod_X[__j]=XX
                    __wi=__i*self.pre.N+__j
                    dw=(eta*(XX-xo)*((YY-yo)*((YY-yo)-theta[__i])/theta[__i])-eta*gamma*W[__wi])
                    W[__wi]+=dt*dw


        if self.theta_squares_the_average:
            for __i in range(self.post.N):
                theta[__i]=y_avg[__i]*y_avg[__i]/theta_o
        else:
            for __i in range(self.post.N):
                if use_beta:
                    YY=Y[__i]/scale_y-beta[__i]
                    if YY<0.0:
                        YY=0.0
                else:
                    YY=Y[__i]/scale_y
                theta[__i]+=sim.dt*((1.0/tau_theta)*(YY*YY/theta_o-theta[__i]))
            
        self.apply_weight_limits()
        
cdef class BCM(connection):
    cdef public double a,tau,xo,yo,theta_o,gamma,eta,tau_theta
    cdef public np.ndarray X,Y,theta,y,x
    cdef public object initial_theta_range
    cdef public double time_between_modification,time_to_next_modification
    cdef public object saved_inputs,saved_weights,saved_outputs
    cdef public int save_inputs
    
    cpdef _reset(self):
        self.X=np.zeros(self.pre.N,dtype=float)
        self.Y=np.zeros(self.post.N,dtype=float)
        self.theta=pylab.rand(self.post.N)*(self.initial_theta_range[1]-
                                   self.initial_theta_range[0])+self.initial_theta_range[0]
        self.y=np.zeros(self.post.N,dtype=float)
        self.x=np.zeros(self.pre.N,dtype=float)
        self.time_to_next_modification=self.time_between_modification
        connection._reset(self)

    def __init__(self,neuron pre,neuron post,
                        initial_weight_range=None,
                        state=None,
                        initial_theta_range=None,):
                        
        # initial theta range needs to be defined before connection init                        
        if initial_theta_range is None:
            initial_theta_range=[0.0,0.1]

        self.initial_theta_range=initial_theta_range

        connection.__init__(self,pre,post,initial_weight_range,state)

        self.name='Spiking BCM'
        self.save_inputs=False
        self.saved_inputs=[]
        self.saved_weights=[]
        self.saved_outputs=[]
        
        self.a=1
        self.tau=100
        self.xo=0
        self.yo=0
        self.theta_o=1
        self.gamma=0
        self.eta=4e-06
        self.tau_theta=10000
        self.time_between_modification=-1.0

        self._reset()

    @cython.cdivision(True)
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    cpdef update(self,double t,simulation sim):
        cdef int __i,__j
    
        cdef double *X=<double *>self.X.data
        cdef double *Y=<double *>self.Y.data
        cdef double *W=self.W
        cdef double *theta=<double *>self.theta.data
        cdef double *y=<double *>self.y.data
        cdef double *x=<double *>self.x.data
        cdef double a=self.a
        cdef double tau=self.tau
        cdef double xo=self.xo
        cdef double yo=self.yo
        cdef double theta_o=self.theta_o
        cdef double gamma=self.gamma
        cdef double eta=self.eta
        cdef double tau_theta=self.tau_theta
        cdef double dt

        cdef int modify_now

        cdef int *pre
        cdef int *post   # spikes for pre and post
        cdef int __wi
        
        
        pre=<int *>self.pre.spiking.data
        post=<int *>self.post.spiking.data
    
        for __j in range(self.pre.N):
            x[__j]+=sim.dt*(-x[__j]/tau+a*pre[__j]/sim.dt)
            X[__j]+=sim.dt*((x[__j]-X[__j])/tau)
        for __i in range(self.post.N):
            y[__i]+=sim.dt*(-y[__i]/tau+a*post[__i]/sim.dt)
            Y[__i]+=sim.dt*((y[__i]-Y[__i])/tau)

        modify_now=False
        if self.time_between_modification<0.0:
            modify_now=True
            dt=sim.dt
        elif t>(self.time_to_next_modification-1e-6):
            modify_now=True
            self.time_to_next_modification+=self.time_between_modification
            dt=self.time_between_modification


        if modify_now:
            for __i in range(self.post.N):
                for __j in range(self.pre.N):
                    __wi=__i*self.pre.N+__j
                    W[__wi]+=dt*(eta*(X[__j]-xo)*((Y[__i]-yo)*((Y[__i]-yo)-theta[__i]))-eta*gamma*W[__wi])
            
            self.apply_weight_limits()

            if self.save_inputs:
                self.saved_inputs.append(self.X.copy())
                self.saved_outputs.append(self.Y.copy())
                self.saved_weights.append(self.weights.copy())

        for __i in range(self.post.N):
            theta[__i]+=sim.dt*((1.0/tau_theta)*(Y[__i]*Y[__i]/theta_o-theta[__i]))



cdef class BCM_LawCooper_Offset(connection):
    cdef public double a,tau,xo,theta_o,gamma,eta,tau_thresh
    cdef public np.ndarray X,Y,theta,y,x,yo  # offset for y depends on w
    cdef public object initial_theta_range
    cdef public double time_between_modification,time_to_next_modification

    cpdef _reset(self):
        self.X=np.zeros(self.pre.N,dtype=float)
        self.Y=np.zeros(self.post.N,dtype=float)
        self.theta=pylab.rand(self.post.N)*(self.initial_theta_range[1]-
                                   self.initial_theta_range[0])+self.initial_theta_range[0]

        self.yo=np.zeros(self.post.N,dtype=float)
        self.y=np.zeros(self.post.N,dtype=float)
        self.x=np.zeros(self.pre.N,dtype=float)
        self.time_to_next_modification=self.time_between_modification

        connection._reset(self)

    def __init__(self,neuron pre,neuron post,
                        initial_weight_range=None,
                        state=None,
                        initial_theta_range=None,):
                        
                        
        if initial_theta_range is None:
            initial_theta_range=[0.0,0.1]

        self.initial_theta_range=initial_theta_range

        connection.__init__(self,pre,post,initial_weight_range,state)

        self.name='Spiking BCM LawCooper'
        self.a=1
        self.tau=100
        self.xo=0
        self.theta_o=1
        self.gamma=0
        self.eta=4e-06
        self.tau_thresh=10000
        self.time_between_modification=-1.0
        self._reset()

    @cython.cdivision(True)
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    cpdef update(self,double t,simulation sim):
        cdef int __i,__j
    
        cdef double *X=<double *>self.X.data
        cdef double *Y=<double *>self.Y.data
        cdef double *W=self.W
        cdef double *theta=<double *>self.theta.data
        cdef double *y=<double *>self.y.data
        cdef double *yo=<double *>self.yo.data
        cdef double *x=<double *>self.x.data
        cdef double a=self.a
        cdef double tau=self.tau
        cdef double xo=self.xo
        cdef double theta_o=self.theta_o
        cdef double gamma=self.gamma
        cdef double eta=self.eta
        cdef double tau_thresh=self.tau_thresh
        cdef double dt
        cdef double sum_w
        cdef int modify_now

        cdef int *pre
        cdef int *post   # spikes for pre and post
        cdef int __wi
        
        
        pre=<int *>self.pre.spiking.data
        post=<int *>self.post.spiking.data
    
        for __j in range(self.pre.N):
            x[__j]+=sim.dt*(-x[__j]/tau+a*pre[__j]/sim.dt)
            X[__j]+=x[__j]-xo

        for __i in range(self.post.N):
            y[__i]+=sim.dt*(-y[__i]/tau+a*post[__i]/sim.dt)

            sum_w=0.0
            for __j in range(self.pre.N):
                __wi=__i*self.pre.N+__j
                sum_w+=W[__wi]

            yo[__i]=xo*sum_w
            Y[__i]+=y[__i]-yo[__i]
            if Y[__i]<0.0:
                Y[__i]=0.0

            theta[__i]+=sim.dt*((1.0/tau_thresh)*(Y[__i]*Y[__i]/theta_o-theta[__i]))


        modify_now=False
        if self.time_between_modification<0.0:
            modify_now=True
            dt=sim.dt
        elif t>(self.time_to_next_modification-1e-6):
            modify_now=True
            self.time_to_next_modification+=self.time_between_modification
            dt=self.time_between_modification

        if modify_now:
            for __i in range(self.post.N):
                for __j in range(self.pre.N):
                    __wi=__i*self.pre.N+__j
                    W[__wi]+=dt*(eta*X[__j]*(Y[__i]*(Y[__i]-theta[__i])/theta[__i])-eta*gamma*W[__wi])
            
            self.apply_weight_limits()
                 


cdef class BCM_TwoThreshold(connection):
    cdef public double tau_x,tau_y,xo,yo,theta_o,gamma,eta,theta_L
    cdef public double y_min,y_max
    cdef public tau_theta,tau_beta
    cdef public double ax,ay
    cdef public bint smoothed_x,smoothed_y
    cdef public np.ndarray theta,beta,y,x,y_avg,X,Y,mod_X,mod_Y,mod_theta,mod_beta
    cdef public object initial_theta_range
    cdef public double time_between_modification,time_to_next_modification
    cdef public int theta_squares_the_average




    cpdef _reset(self):
        self.theta=pylab.rand(self.post.N)*(self.initial_theta_range[1]-
                                   self.initial_theta_range[0])+self.initial_theta_range[0]
        self.beta=pylab.rand(self.post.N)*(self.initial_theta_range[1]-
                                   self.initial_theta_range[0])+self.initial_theta_range[0]

        self.mod_theta=np.zeros(self.post.N,dtype=float)
        self.mod_beta=np.zeros(self.post.N,dtype=float)


        self.y=np.zeros(self.post.N,dtype=float)
        self.Y=np.zeros(self.post.N,dtype=float)
        self.mod_Y=np.zeros(self.post.N,dtype=float)

        self.y_avg=np.sqrt(self.theta)
        self.x=np.zeros(self.pre.N,dtype=float)
        self.X=np.zeros(self.pre.N,dtype=float)
        self.mod_X=np.zeros(self.pre.N,dtype=float)

        self.time_to_next_modification=self.time_between_modification

        connection._reset(self)

    def __init__(self,neuron pre,neuron post,
                        initial_weight_range=None,
                        state=None,
                        initial_theta_range=None,):
                        
                        
        if initial_theta_range is None:
            initial_theta_range=[0.0,0.1]

        self.initial_theta_range=initial_theta_range

        connection.__init__(self,pre,post,initial_weight_range,state)

        self.name='Spiking BCM TwoThreshold'
        self.tau_x=0.01
        self.tau_y=0.01
        self.ax=1.0/self.tau_x
        self.ay=1.0/self.tau_y
        self.smoothed_x=False
        self.smoothed_y=False

        self.xo=0
        self.yo=0
        self.y_min=-1e500
        self.y_max=1e500
        self.theta_o=1
        self.theta_L=0.0
        self.gamma=0
        self.eta=4e-06
        self.tau_theta=10
        self.tau_beta=-1
        self.time_between_modification=-1.0
        self.theta_squares_the_average=False
        self._reset()

        self.save_attrs.extend(['tau_x','tau_y','xo','yo','theta_o',
            'gamma','eta','tau_theta','tau_beta','ax',
            'ay','smoothed_x','smoothed_y','time_between_modification',
            'time_to_next_modification','theta_squares_the_average','theta_L'])
        self.save_data.extend(['theta','beta','x','y',
            'y_avg','X','Y','mod_X','mod_Y','mod_theta','mod_beta',
            'initial_theta_range'])



    @cython.cdivision(True)
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    cpdef update(self,double t,simulation sim):
        cdef int __i,__j
    
        cdef double *W=self.W
        cdef double *theta=<double *>self.theta.data
        cdef double *beta=<double *>self.beta.data
        cdef double *y=<double *>self.y.data
        cdef double *x=<double *>self.x.data
        cdef double *Y=<double *>self.Y.data
        cdef double *X=<double *>self.X.data
        cdef double *y_avg=<double *>self.y_avg.data

        cdef double *mod_X=<double *>self.mod_X.data
        cdef double *mod_Y=<double *>self.mod_Y.data
        cdef double *mod_theta=<double *>self.mod_theta.data
        cdef double *mod_beta=<double *>self.mod_beta.data

        cdef double tau_x=self.tau_x
        cdef double tau_y=self.tau_y
        cdef double ax=self.ax
        cdef double ay=self.ay
        cdef double xo=self.xo
        cdef double yo=self.yo
        cdef double theta_o=self.theta_o
        cdef double theta_L=self.theta_L
        cdef double gamma=self.gamma
        cdef double eta=self.eta
        cdef double tau_theta=self.tau_theta
        cdef double tau_beta=self.tau_beta
        cdef double dt,dw
        cdef double YY
        cdef int modify_now,use_beta

        cdef int *pre
        cdef int *post   # spikes for pre and post
        cdef int __wi
        
        use_beta=self.tau_beta>0

        pre=<int *>self.pre.spiking.data
        post=<int *>self.post.spiking.data
    

        for __j in range(self.pre.N):
            x[__j]+=sim.dt*(-x[__j]/tau_x+ax*pre[__j]/sim.dt)

        if self.smoothed_x:
            for __j in range(self.pre.N):
                X[__j]+=sim.dt*(x[__j]-X[__j])/tau_x
        else:
            for __j in range(self.pre.N):
                X[__j]=x[__j]

        for __i in range(self.post.N):
            y[__i]+=sim.dt*(-y[__i]/tau_y+ay*post[__i]/sim.dt)

        if self.smoothed_y:
            for __i in range(self.post.N):
                Y[__i]+=sim.dt*(y[__i]-Y[__i])/tau_y
        else:
            for __i in range(self.post.N):
                Y[__i]=y[__i]


        for __i in range(self.post.N):
            y_avg[__i]+=sim.dt*((1.0/tau_theta)*((Y[__i]/tau_y-yo)-y_avg[__i]))            

        if use_beta:
            for __i in range(self.post.N):
                beta[__i]+=sim.dt*(1.0/tau_beta)*(Y[__i]-beta[__i])

        modify_now=False
        if self.time_between_modification<0.0:
            modify_now=True
            dt=sim.dt
        elif t>(self.time_to_next_modification-1e-6):
            modify_now=True
            self.time_to_next_modification+=self.time_between_modification
            dt=self.time_between_modification

        if modify_now:
            for __i in range(self.post.N):
                if use_beta:
                    YY=Y[__i]-beta[__i]
                    if YY<0:
                        YY=0

                else:
                    YY=Y[__i]

                if YY<self.y_min:
                    YY=self.y_min
                elif YY>self.y_max:
                    YY=self.y_max

                mod_Y[__i]=YY
                mod_theta[__i]=theta[__i]
                mod_beta[__i]=beta[__i]


                for __j in range(self.pre.N):
                    mod_X[__j]=X[__j]
                    __wi=__i*self.pre.N+__j
                    dw=(eta*(X[__j]-xo)*((YY-yo)*((YY-yo)-theta[__i])/theta[__i])-eta*gamma*W[__wi])
                    W[__wi]+=dt*dw


        if self.theta_squares_the_average:
            for __i in range(self.post.N):
                theta[__i]=y_avg[__i]*y_avg[__i]/theta_o
        else:
            for __i in range(self.post.N):
                if use_beta:
                    YY=Y[__i]-beta[__i]
                    if YY<0.0:
                        YY=0.0
                else:
                    YY=Y[__i]
                theta[__i]+=sim.dt*((1.0/tau_theta)*(YY*YY/theta_o-theta[__i]))
            
        self.apply_weight_limits()
