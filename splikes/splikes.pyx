version='0.0.8'

cimport cython

import numpy as np
cimport numpy as np

from copy import deepcopy
import pylab
from Waitbar import Waitbar

cdef extern from "math.h":
    double sqrt(double)
    double exp(double)
    double log(double)
    double tanh(double)
    double pow(double,double)


cdef  rk_state global_state

cdef  init_by_int(int seed):
    rk_seed(seed, &global_state)

cdef  init_by_entropy():
    rk_randomseed(&global_state)

cdef double randu():
    return(rk_double(&global_state))

cdef double randn():
    return(rk_gauss(&global_state))

cdef double rande():
    cdef double y
    y=2.0*randu()-1.0
    if y<0.0:
        return log(-y)
    elif y>0.0:
        return -log(y)
    else:
        return 0.0

import time
def time2str(tm):
    if tm<0:
        return None

    frac=tm-int(tm)
    tm=int(tm)
    
    s=''
    sc=tm % 60
    tm=tm//60
    
    mn=tm % 60
    tm=tm//60
    
    hr=tm % 24
    tm=tm//24
    dy=tm

    if (dy>0):
        s=s+"%d d, " % dy

    if (hr>0):
        s=s+"%d h, " % hr

    if (mn>0):
        s=s+"%d m, " % mn


    s=s+"%.2f s" % (sc+frac)

    return s

import sys

from copy import deepcopy
import pylab

ms=0.001
second=1000*ms
minute=60*second
hour=60*minute
day=24*hour
year=365.25*day
Hz=1.0

cdef class group:

    def save(self,g):
        if self.verbose:
            print str(type(self)),":",str(self.__getattribute__('name'))
            sys.stdout.flush()


        g.attrs['type']=str(type(self))
        g.attrs['name']=str(self.__getattribute__('name'))


        for attr in self.save_attrs:
            if self.verbose:
                print "\t",attr
                sys.stdout.flush()
            g.attrs[attr]=self.__getattribute__(attr)

        for dataname in self.save_data:
            if self.verbose:
                print "\t",dataname
                sys.stdout.flush()
            data=self.__getattribute__(dataname)

            if self.verbose:
                print data
                sys.stdout.flush()

            if data is None:
                if self.verbose:
                    print "(skipping)"
                    sys.stdout.flush()
                continue

            g.create_dataset(dataname,data=data)





cdef class monitor(group):
    
    def __init__(self,container,name,save_interval,start_time=0.0):
        self.name=name
        self.container=container
        self.time_to_next_save=start_time
        self.save_interval=save_interval
        self._reset()

        self.save_attrs=['time_to_next_save','save_interval']
        self.save_data=['t','values']

    def save(self,g):
        # to save the values, we need to make them arrays
        # if we want to continue a simulation, we need them to stay as lists
        self.t_tmp,self.values_tmp=self.t,self.values

        self.t=np.array(self.t)
        self.values=np.array(self.values).squeeze()

        group.save(self,g)
        self.t,self.values=self.t_tmp,self.values_tmp

    def _reset(self):
        self.t=[]
        self.values=[]
        
    cpdef update(self,double t):
        if t<=self.time_to_next_save:
            return
        self.t.append(t)
        variable=self.container.__getattribute__(self.name)
        self.values.append(deepcopy(variable))
        self.time_to_next_save+=self.save_interval




    def arrays(self):
        return self.time_array(),self.array()
        
    def array(self):
        return np.array(self.values).squeeze()
        
    def time_array(self):
        return np.array(self.t)
        
    def plot(self,*args,**kwargs):
        import matplotlib.pyplot as plt 
        import matplotlib.ticker 
    
        def HMSFormatter(value, loc): 
            h = value // 3600 
            m = (value - h * 3600) // 60 
            s = value % 60 
            return "%02d:%02d:%02d" % (h,m,s) 
    
        def HMSFormatter2(value, loc): 
            h = value // 3600 
            m = (value - h * 3600) // 60 
            s = value % 60 
            ms=value%1
        
            return "%02d:%02d.%03d" % (m,s,ms*1000) 

        t,y=self.arrays()
        if np.max(t)<10:  # use ms
            pylab.gca().xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(HMSFormatter2)) 
        else:
            pylab.gca().xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(HMSFormatter)) 

        pylab.plot(t,y,*args,**kwargs) 
        pylab.gcf().autofmt_xdate()

        pylab.ylabel(self.name)


cdef class simulation(group):
    
    def __init__(self,total_time,dt=0.00025,start_time=0.0):
        self.dt=dt
        self.total_time=total_time
        self.start_time=start_time
        self.time_to_next_save=1e500
        self.time_to_next_filter=1e500
        self.seed=-1
        self.monitors={}
        self.filters=[]  # functions for processing
        self.save_attrs=['seed','total_time','dt','time_to_next_save','time_to_next_filter',
                    'verbose',]
        self.save_data=[]
        self.verbose=False
        self.name='simulation'

    cpdef _reset(self):
        if self.seed<0:
            init_by_entropy()
            pylab.seed(None)
        else:
            init_by_int(self.seed)
            pylab.seed(self.seed)
       
    def add_filter(self,function,time):
        self.filters.append( {'function':function,
                              'interval':time,
                              'time_to_next':time} )

        self.time_to_next_filter=min([x['time_to_next'] for x in self.filters])
            
    def monitor(self,container,names,save_interval,start_time=0.0):
        if isinstance(names,str):
            names=[names]
        
        for varname in names:
            if varname in self.monitors:
                objname=container.name
                if objname is None:
                    objname=str(container)
                    
                name='%s [%s]' % (varname,objname)
            else:
                name=varname
                
            self.monitors[name]=monitor(container,varname,save_interval,start_time)
            
        self.time_to_next_save=min([self.monitors[name].time_to_next_save for name in self.monitors])
        
        
cdef class neuron(group):
    
    def __init__(self,N):
        self.N=N
        self.num_pre=0
        self.num_post=0
        self.verbose=False
        self.spiking=np.zeros(self.N,dtype=np.int32)    
        self.last_spike_time=-np.ones(self.N,dtype=np.float)    
        self.save_spikes_begin=0.0
        self.save_spikes_end=-1.0
        self.connections_pre=[]
        self.connections_post=[]
        self.name=None
    
        self.save_attrs=['num_pre','num_post','N','verbose',
                            'save_spikes_begin','save_spikes_end','post_count']
        self.save_data=['spiking','last_spike_time','rate','saved_spikes']



    cpdef _reset(self):
        self.spiking=np.zeros(self.N,dtype=np.int32)            
        self.last_spike_time=-np.ones(self.N,dtype=np.float)    
        self.rate=np.zeros(self.N,dtype=np.float)
        self.saved_spikes=[]
        self.is_spike=0
        self.post_count=0
        
    @cython.cdivision(True)
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    cpdef update(self,double t,simulation sim):
        pass
        
        
    def plot_spikes(self,neuron_offset=0):
        t,n=zip(*self.saved_spikes)
        t=np.array(t)
        n=np.array(n)
        
        pylab.plot(t,n+neuron_offset,'.')
        pylab.ylabel('neuron')
        pylab.xlabel('time')
        
        yl=[min(n)-1,max(n)+1]
        pylab.gca().set_ylim(yl)
        pylab.gca().set_yticks(range(max(n)+neuron_offset+2))
        
        pylab.draw()
        


        
cdef class connection(group):
    
    cpdef _reset(self):
        if self.reset_to_initial:
            self.weights=self.initial_weights.copy()
        else:
            self.weights=pylab.rand(self.post.N,self.pre.N)*(self.initial_weight_range[1]-
                                       self.initial_weight_range[0])+self.initial_weight_range[0]

        self.W=<double *>self.weights.data
        self.initial_weights=self.weights.copy()
        
        if self.use_state:
            self.state=self.post.__getattribute__(self.state_variable)
    



    def __init__(self,neuron pre,neuron post,initial_weight_range=None,state=None):
        cdef np.ndarray arr
    
        if initial_weight_range is None:
            initial_weight_range=[.00095,.00105]
            
        self.initial_weight_range=initial_weight_range
        self.pre=pre
        self.post=post
        self.post.connections_pre.append(self)
        self.pre.connections_post.append(self)
        self.w_max=1e500
        self.w_min=-1e500
        self.spike_scale=1.0
        self.reset_to_initial=False
        self.name=None
        
        if state is None:
            self.use_state=False
            self.state_variable=None
        else:
            self.use_state=True
            self.state_variable=state
            
        self.save_attrs=['verbose','reset_to_initial','w_min','w_max',
                    'spike_scale','use_state']
        self.save_data=['initial_weight_range','initial_weights','weights','state_variable']

        self._reset()
    
    cpdef apply_weight_limits(self):
        cdef int __i
        cdef int __j
        cdef int __wi
        cdef double *W=self.W
        
        for __i in range(self.post.N):
            for __j in range(self.pre.N):
                __wi=__i*self.pre.N+__j
                if W[__wi]<self.w_min:
                    W[__wi]=self.w_min

                if W[__wi]>self.w_max:
                    W[__wi]=self.w_max            
                        
                        
    @cython.cdivision(True)
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    cpdef update(self,double t,simulation sim):
        pass
    


def run_sim(simulation sim,object neurons,object connections,
                    int display_hash=False,int print_time=True,
                    object display=None,double time_between_display=1.0):
    
    cdef double t=sim.start_time
    cdef double hash_step,next_hash
    cdef double next_display=sim.start_time+time_between_display
    cdef double duration=sim.total_time-sim.start_time
    cdef int i,j,num_neurons,num_connections
    cdef int use_display
    hash_step=duration/100
    next_hash=sim.start_time

    if display is None:
        use_display=False
    else:
        use_display=True

    sim._reset()
    num_neurons=len(neurons)
    num_connections=len(connections)
    
    for i in range(num_neurons):
        neurons[i]._reset()
    
    for i in range(num_connections):
        connections[i]._reset()

    if display_hash:
         wb = Waitbar(False)
    
    if print_time:
        t1=time.time()
        
    for name in sim.monitors:
        sim.monitors[name].update(t)
    
    while t<=sim.total_time:
        for i in range(num_neurons):
            neurons[i].update(t,sim)
        
            #pre.update(t,sim,post,c)
        for i in range(num_connections):
            connections[i].update(t,sim)
        t+=sim.dt

        if t>=sim.time_to_next_filter:
            for filter in sim.filters:
                if t>filter['time_to_next']:
                    filter['function'](t,sim,neurons,connections)
                    filter['time_to_next']+=filter['interval']
            
            sim.time_to_next_filter=min([x['time_to_next'] 
                                                for x in sim.filters])

        if t>=sim.time_to_next_save:
            for name in sim.monitors:
                sim.monitors[name].update(t)
            sim.time_to_next_save=min([sim.monitors[name].time_to_next_save 
                                                for name in sim.monitors])

        if use_display and t>next_display:
            display(t,sim,neurons,connections)
            next_display+=time_between_display


        if display_hash and t>next_hash:
            wb.updated((t-sim.start_time)/duration)
            next_hash+=hash_step

    if print_time:
        print "Time Elapsed...",time2str(time.time()-t1)
        
 