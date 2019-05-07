cimport cython

import numpy as np
cimport numpy as np

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import h5py

inf=1e500
import sys

from copy import deepcopy
import pylab
from Waitbar import Waitbar
from copy import deepcopy

def dot(what="."):
    import sys
    sys.stdout.write(what)
    sys.stdout.flush()

cdef extern from "math.h":
    double sqrt(double)
    double exp(double)
    double log(double)
    double tanh(double)
    double pow(double,double)


cdef  rk_state global_state

cpdef  init_by_int(int seed):
    rk_seed(seed, &global_state)

cpdef  init_by_entropy():
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

cdef class group:

    def save(self,g):
        if self.verbose:
            print(str(type(self)),":",str(self.__getattribute__('name')))
            sys.stdout.flush()


        g.attrs['type']=str(type(self))
        g.attrs['name']=str(self.__getattribute__('name'))


        for attr in self.save_attrs:
            if self.verbose:
                print("\t",attr)
                sys.stdout.flush()
            g.attrs[attr]=self.__getattribute__(attr)

        for dataname in self.save_data:
            if self.verbose:
                print("\t",dataname)
                sys.stdout.flush()
            data=self.__getattribute__(dataname)

            if self.verbose:
                print(data)
                sys.stdout.flush()

            if data is None:
                if self.verbose:
                    print("(skipping)")
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

    def _reset(self):
        self.t=[]
        self.values=[]

    def save(self,g):
        # to save the values, we need to make them arrays
        # if we want to continue a simulation, we need them to stay as lists
        self.t_tmp,self.values_tmp=self.t,self.values

        self.t=np.array(self.t)
        self.values=np.array(self.values).squeeze()

        group.save(self,g)
        self.t,self.values=self.t_tmp,self.values_tmp

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

            d=h//24
            h=h%24
            if d==0:
                return "%02d:%02d:%02d" % (h,m,s) 
            else:
                return "%dd %02d:%02d:%02d" % (d,h,m,s) 
    
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
    
    def __init__(self,total_time,dt=1.0,start_time=0.0):
        self.dt=dt
        self.total_time=total_time
        self.start_time=start_time
        self.current_time=start_time
        self.time_to_next_save=1e500
        self.time_to_next_filter=1e500
        self.seed=-1
        self.monitors={}
        self.post_process=[]
        self.verbose=False
        self.filters=[]  # functions for processing
        self.save_attrs=['seed','total_time','dt','time_to_next_save','time_to_next_filter',
                    'verbose',]
        self.save_data=[]
        self.name='simulation'
        
    cpdef _reset(self):
        if self.seed<0:
            init_by_entropy()
            pylab.seed(None)
        else:
            init_by_int(self.seed)
            pylab.seed(self.seed)
            
            
        for name in self.monitors:
            self.monitors[name]._reset()
            
            
    def add_filter(self,function,time):
        self.filters.append( {'function':function,
                              'interval':time,
                              'time_to_next':time} )

        self.time_to_next_filter=min([x['time_to_next'] for x in self.filters])

    def monitor(self,container,names,save_interval,start_time=0.0):
        if isinstance(names,str):
            names=[names]
        
        for name in names:
            i=1
            original_name=name
            while name in self.monitors:
                name=original_name+'_%d' % i
                i+=1
        
            self.monitors[name]=monitor(container,original_name,save_interval,start_time)
            
        self.time_to_next_save=min([self.monitors[name].time_to_next_save for name in self.monitors])
        
    def save(self,g):
        group.save(self,g)

        for i,p in enumerate(self.post_process):
            g2=g.create_group("process %d" % i)
            p.save(g2)

    def __iadd__(self,other): # add some post-processing
        self.post_process.append(other)
        other.sim=self
        return self

        
cdef class neuron(group):
    
    def __init__(self,N):
        self.N=N
        self.num_pre=0
        self.num_post=0
        self.verbose=False
        self.connections_pre=[]
        self.connections_post=[]
        self.post_process=[]
        self.output=np.zeros(self.N,dtype=np.float)
        self.linear_output=np.zeros(self.N,dtype=np.float)

        self.save_attrs=['num_pre','num_post','verbose',]
        self.save_data=['output','linear_output']

        self.name=None

    def save(self,g):
        group.save(self,g)

        for i,p in enumerate(self.post_process):
            g2=g.create_group("process %d" % i)
            p.save(g2)

    cpdef _reset(self):
        self.linear_output=np.zeros(self.N,dtype=np.float)
        self.output=np.zeros(self.N,dtype=np.float)    

    cpdef _clean(self):
        pass

    @cython.cdivision(True)
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    cpdef update(self,double t,simulation sim):
        cdef int i,j
        cdef double *y=<double *>self.linear_output.data
        cdef double *z=<double *>self.output.data
        
        if self.verbose:
            dot("[neuron in update]")
        for i in range(self.N):
            y[i]=0.0
            z[i]=0.0
        if self.verbose:
            dot("[neuron out update]")


    def __add__(self,other):  # make a channel
        try:
            neuron_list=self.nlist+other.nlist
        except AttributeError:
            neuron_list=[self]+[other]

        return channel(neuron_list)

    def __iadd__(self,other): # add some post-processing
        self.post_process.append(other)
        other.n=self
        return self


cdef class channel(neuron):

    cpdef _reset(self):
        cdef int N,k
            
        N=0
        for k in range(self.number_of_neurons):
            self.neuron_list[k]._reset()
        
            self.linear_output[N:(N+self.neuron_list[k].N)]=self.neuron_list[k].linear_output[:]
            self.output[N:(N+self.neuron_list[k].N)]=self.neuron_list[k].output[:]
            
            self.neuron_list[k].linear_output=self.linear_output[N:(N+self.neuron_list[k].N)]
            self.neuron_list[k].output=self.output[N:(N+self.neuron_list[k].N)]
            N+=self.neuron_list[k].N
    
    cpdef _clean(self):
        cdef int N,k
        for k in range(self.number_of_neurons):
            self.neuron_list[k]._clean()

    
    def __init__(self,nlist,verbose=False):
        cdef int k,N
        
        self.neuron_list=nlist
        self.number_of_neurons=len(self.neuron_list)
        self.post_process=[]
        
        N=0
        for k in range(self.number_of_neurons):
            if self.verbose:
                dot("[channel %d in update]" % k)

            N+=self.neuron_list[k].N
            
        neuron.__init__(self,N) 
        self.name=None
        
        self.save_attrs.extend(['number_of_neurons'])

        self._reset()
        
        self.post_process.append(post_process_channel(self))
        
    @cython.cdivision(True)
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    cpdef update(self,double t,simulation sim):
        cdef int k,N
        
        for k in range(self.number_of_neurons):
            if self.verbose:
                dot("o")
            self.neuron_list[k].update(t,sim)
        
    def __getitem__(self,index):
        return self.neuron_list[index]    
        
    def __len__(self):
         return len(self.neuron_list)    
    
cdef class post_process_neuron(group):
    cpdef _reset(self):
        pass
        
    def __init__(self):
        self.save_attrs=[]
        self.save_data=[]
    
    @cython.cdivision(True)
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    cpdef update(self,double t,simulation sim):
        pass
        
        
        
cdef class post_process_channel(group):
    
    def __init__(self,channel ch):
        self.ch=ch
        self.save_attrs=[]
        self.save_data=[]
        
    cpdef _reset(self):
        cdef int num_neurons,i,L,k
        
        num_neurons=len(self.ch.neuron_list)
    
        for i in range(num_neurons):
            L=len(self.ch.neuron_list[i].post_process)
            for k in range(L):
                self.ch.neuron_list[i].post_process[k]._reset()
                
    @cython.cdivision(True)
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    cpdef update(self,double t,simulation sim):
        cdef int num_neurons,i,L,k
        
        num_neurons=len(self.ch.neuron_list)
    
        for i in range(num_neurons):
            if self.ch.verbose:
                dot()
            L=len(self.ch.neuron_list[i].post_process)
            for k in range(L):
                if self.ch.verbose:
                    dot('X')
                self.ch.neuron_list[i].post_process[k].update(t,sim)
            
        
cdef class connection(group):

    cpdef _reset(self):
        if self.reset_to_initial:
            self.weights=self.initial_weights.copy()
        else:
            self.weights=pylab.rand(self.post.N,self.pre.N)*(self.initial_weight_range[1]-
                                       self.initial_weight_range[0])+self.initial_weight_range[0]

        self.w=<double *>self.weights.data
        self.initial_weights=self.weights.copy()

    cpdef _clean(self):
        pass


    def __init__(self,neuron pre,neuron post,initial_weight_range=None):
        cdef np.ndarray arr
    
        if initial_weight_range is None:
            initial_weight_range=pylab.array([.00095,.00105])
        else:
            initial_weight_range=pylab.array(initial_weight_range)
            
        self.reset_to_initial=False    
        self.initial_weight_range=initial_weight_range
        self.pre=pre
        self.post=post
        self.post.connections_pre.append(self)
        self.pre.connections_post.append(self)
        self.post_process=[]
        self.name=None
        self.verbose=False
        self.save_attrs=['verbose','reset_to_initial']
        self.save_data=['initial_weight_range','initial_weights','weights']
        self._reset()
    
                        
    @cython.cdivision(True)
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    cpdef update(self,double t,simulation sim):
        pass

    def __iadd__(self,other): # add some post-processing
        self.post_process.append(other)
        other.c=self
        return self



cdef class post_process_connection(group):
    cpdef _reset(self):
        pass

    def __init__(self,):
        self.save_attrs=[]
        self.save_data=[]
    
    @cython.cdivision(True)
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    cpdef update(self,double t,simulation sim):
        pass
        
        
def run_sim(simulation sim,object neurons,object connections,
                    int display_hash=False,int print_time=True,debug=False,
                    object display=None,double time_between_display=1.0):
    
    cdef double t=sim.start_time
    cdef double next_display=sim.start_time+time_between_display    
    cdef int _debug=debug
    cdef double hash_step,next_hash
    cdef double t1,t2,next_hash_time
    cdef double duration=sim.total_time
    cdef double end_time=sim.start_time+sim.total_time
    cdef int i,j,num_neurons,num_connections
    cdef int use_display


    if display is None:
        use_display=False
    else:
        use_display=True


    hash_step=duration/100
    next_hash=sim.start_time

    cdef int L,k

    sim._reset()
    num_neurons=len(neurons)
    num_connections=len(connections)
    
    _debug=_debug or sim.verbose

    for i in range(num_neurons):
        if _debug:
            dot('[init neuron %d]' % i)

        neurons[i]._reset()
        L=len(neurons[i].post_process)
        for k in range(L):
            neurons[i].post_process[k]._reset()
    
    
    # it makes sense for the connections to do the post-process after the reset
    # to make sure the initial weights are not out of bounds, not normalized, etc...
    # this way we can enforce the zero diagonal as well, right from the beginning
    
    for i in range(num_connections):
        if _debug:
            dot('[init conn %d]' % i)

        connections[i]._reset()

        L=len(connections[i].post_process)
        for k in range(L):
            connections[i].post_process[k]._reset()

        for k in range(L):
            connections[i].post_process[k].update(t-sim.dt,sim)


    if display_hash:
         wb = Waitbar(False)
    
    if print_time:
        t1=time.time()
        
    next_hash_time=time.time()+1

    for name in sim.monitors:
        if _debug:
            dot("[init monitor %s]" % name)    

        sim.monitors[name].update(t)

    while t<=end_time:
        if _debug:
            dot("[t %f]" % t)    

        for i in range(num_neurons):
            if _debug:
                dot('[update neuron %d]' % i)

            neurons[i].update(t,sim)
            if _debug:
                dot('[post process neuron %d]' % i)
                
            L=len(neurons[i].post_process)
            for k in range(L):
                neurons[i].post_process[k].update(t,sim)

        
        for i in range(num_connections):
            if _debug:
                dot('[update conn %d]' % i)

            connections[i].update(t,sim)
            
            L=len(connections[i].post_process)
            for k in range(L):
                connections[i].post_process[k].update(t,sim)
            
            
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
            t2=time.time()
            if t2>next_hash_time:
                wb.updated((t-sim.start_time)/duration)
                next_hash_time=t2+1
                
            next_hash+=hash_step

        sim.current_time=t


    L=len(sim.post_process)
    for k in range(L):
        sim.post_process[k].apply()


    for i in range(num_neurons):
        neurons[i]._clean()
    for i in range(num_connections):
        connections[i]._clean()

    if print_time:
        print("Sim Time Elapsed...%s" % time2str(time.time()-t1))
        
 