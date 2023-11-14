from splikes.splikes cimport *
cimport cython
import matplotlib.pyplot as pylab
#import plasticnet as pn

import numpy as np
cimport numpy as np
from splikes.neurons.isi_distributions cimport *

cdef class input_current(neuron):
    cdef public int pattern_number
    cdef public np.ndarray current_values
    cdef public np.ndarray current
    cdef public int number_of_patterns
    cdef public double time_between_patterns,time_to_next_pattern

    cpdef _reset(self):
        neuron._reset(self)
        self.time_to_next_pattern=0.0 
        self.pattern_number=-1      


    def __init__(self,current_values,time_between_patterns=0.2,shape=None,verbose=False):
        self.current_values=np.ascontiguousarray(np.atleast_2d(np.array(current_values,float)))

        assert self.current_values.ndim==2,"current_values array must be 2D"

        if not shape is None:
            self.current_values=self.current_values.reshape(shape)
            
        neuron.__init__(self,self.current_values.shape[1]) # number of neurons
        self.number_of_patterns=self.current_values.shape[0]
        self.time_between_patterns=time_between_patterns
        self.verbose=verbose
        self.name='Input Current'

        self.use_I=True

        self.save_attrs.extend(['number_of_patterns','time_between_patterns',])
        self.save_data.extend(['current_values',])

        self._reset()
        

    cpdef new_current(self,double t):

        self.pattern_number+=1
        if self.pattern_number>=self.number_of_patterns:
            self.pattern_number=0
                
        self.current=self.current_values[self.pattern_number]

        self.time_to_next_pattern=t+self.time_between_patterns
        if self.verbose:
            print("New pattern %d" % self.pattern_number)
            self.print_current_values()
            print("Time to next current_values: %f" % self.time_to_next_pattern)
        
        cdef int i
        cdef double *current=<double *>self.current.data
        cdef double *Ic=<double *>self.Ic.data
        for i in range(self.N):
            Ic[i]=current[i]



    def print_current(self):
        cdef int i
        cdef double *current=<double *>self.current.data
        for i in range(self.N):
            print(current[i])
            

    @cython.cdivision(True)
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    cpdef update(self,double t,simulation sim):
        cdef double r
        cdef int i,j
        cdef double *Ic=<double *>self.Ic.data
        
        cdef double *current

        if t>=(self.time_to_next_pattern-1e-6):  # the 1e-6 is because of binary represenation offsets
            self.new_current(t)



cdef class poisson_pattern(neuron):
    cdef public int sequential
    cdef public int pattern_number
    cdef public np.ndarray patterns
    cdef public np.ndarray pattern
    cdef public int number_of_patterns
    cdef public double time_between_patterns,time_to_next_pattern
    
    cpdef _reset(self):
        neuron._reset(self)
        self.time_to_next_pattern=0.0 
        self.pattern_number=-1      
        
    def __init__(self,patterns,time_between_patterns=0.2,sequential=False,shape=None,verbose=False):
        self.patterns=np.ascontiguousarray(np.atleast_2d(np.array(patterns,float)))

        assert self.patterns.ndim==2,"pattern array must be 2D"

        if not shape is None:
            self.patterns=self.patterns.reshape(shape)
            
        self.sequential=sequential
        neuron.__init__(self,self.patterns.shape[1]) # number of neurons
        self.number_of_patterns=self.patterns.shape[0]
        self.time_between_patterns=time_between_patterns
        self.verbose=verbose
        self.name='Poisson Pattern'
        
        self.save_attrs.extend(['number_of_patterns','time_between_patterns','sequential'])
        self.save_data.extend(['patterns','pattern'])



        self._reset()
        
        
    def plot_spikes(self,count=False):
        spikes=self.saved_spikes
        t=[x[0] for x in spikes]
        n=[x[1] for x in spikes]

        neuron.plot_spikes(self,count)
        yl=[min(n)-1,max(n)+1]
        pylab.gca().set_ylim(yl)
        pylab.gca().set_yticks(range(max(n)+2))
        # tt=0
        # while tt<max(t):
        #     pylab.plot([tt,tt],yl,'c:',lw=0.5)
        #     if count:
                
        #         for nn in range(max(n)+1):
        #             c=len([x for _t,_n in zip(t,n) if 
        #                         tt<=_t<tt+self.time_between_patterns and _n==nn])
        #             pylab.text(tt+self.time_between_patterns/2.0,nn+0.1,'%d' % c)
                
        #         tt+=self.time_between_patterns
        # pylab.draw()

    cpdef new_pattern(self,double t):
        if not self.sequential:
            if self.verbose:
                print("random")
            self.pattern_number=<int> (randu()*self.number_of_patterns)
        else:
            if self.verbose:
                print("sequential")
            self.pattern_number+=1
            if self.pattern_number>=self.number_of_patterns:
                self.pattern_number=0
                
        self.pattern=self.patterns[self.pattern_number]

        self.time_to_next_pattern=t+self.time_between_patterns
        if self.verbose:
            print("New pattern %d" % self.pattern_number)
            self.print_pattern()
            print("Time to next pattern: %f" % self.time_to_next_pattern)
        
        cdef int i
        cdef double *pattern=<double *>self.pattern.data
        for i in range(self.N):
            self.rate[i]=pattern[i]

    def print_pattern(self):
        cdef int i
        cdef double *pattern=<double *>self.pattern.data
        for i in range(self.N):
            print(pattern[i])
            

    @cython.cdivision(True)
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    cpdef update(self,double t,simulation sim):
        cdef double r
        cdef int i,j
        cdef double *rate=<double *>self.rate.data
        
        cdef double *pattern
        cdef int *spiking=<int *>self.spiking.data
        
        if t>=(self.time_to_next_pattern-1e-6):  # the 1e-6 is because of binary represenation offsets
            self.new_pattern(t)
        pattern=<double *>self.pattern.data    
        
        self.is_spike=0
        for i in range(self.N):
            r=randu()
            if r<(pattern[i]*sim.dt):
                self.is_spike=1
                spiking[i]=1
                if self.save_spikes_begin<=t<=self.save_spikes_end:
                    self.saved_spikes.append( (t,i) )
            else:
                spiking[i]=0
  
cdef class poisson_plasticnet(neuron):
    cdef public object pneuron,psim
    cdef public double time_between_patterns,time_to_next_pattern
    #cdef public seed
    
    cpdef _reset(self):
        cdef int L,k

        neuron._reset(self)

        # if self.seed<0:
        #     pn.init_by_entropy()
        # else:
        #     pn.init_by_int(self.seed)

        self.pneuron._reset()
        self.time_to_next_pattern=0.0 
        self.pneuron.time_to_next_pattern=0.0

        L=len(self.pneuron.post_process)
        for k in range(L):
            self.pneuron.post_process[k]._reset()


    def __init__(self,pneuron,sim=None,time_between_patterns=0.2,verbose=False):
        self.pneuron=pneuron
        self.psim=sim

        neuron.__init__(self,self.pneuron.N) # number of connections
        self.time_between_patterns=time_between_patterns
        self.pneuron.time_between_patterns=time_between_patterns

        self.verbose=verbose
        
        self.name='Poisson Plasticnet'
        
        self._reset()

        self.save_attrs.extend(['time_between_patterns'])

    def save(self,g):

        group.save(self,g)

        g2=g.create_group("pneuron")
        self.pneuron.save(g2)

        if not self.psim is None:
            g2=g.create_group("psim")
            self.psim(g2)

    def plot_spikes(self,count=False):
        spikes=self.saved_spikes
        t=[x[0] for x in spikes]
        n=[x[1] for x in spikes]

        neuron.plot_spikes(self,count)
        yl=[min(n)-1,max(n)+1]
        pylab.gca().set_ylim(yl)
        pylab.gca().set_yticks(range(max(n)+2))
        tt=0
        while tt<max(t):
            pylab.plot([tt,tt],yl,'c:')
            
            if count:
                for nn in range(max(n)+1):
                    c=0
                    for _t,_n in zip(t,n):
                        if tt<=_t<tt+self.time_between_patterns and _n==nn:
                            c+=1
                    pylab.text(tt+self.time_between_patterns/2.0,nn+0.1,'%d' % c)
                                
            tt+=self.time_between_patterns
        pylab.draw()


    cpdef new_pattern(self,double t):
        cdef int L,k,i

        self.pneuron.update(t,self.psim)
        L=len(self.pneuron.post_process)
        for k in range(L):
            self.pneuron.post_process[k].update(t,self.psim)

        self.time_to_next_pattern=t+self.time_between_patterns
        self.pneuron.time_to_next_pattern=t+self.time_between_patterns

        for i in range(self.N):
            self.rate[i]=self.pneuron.output[i]

        if self.verbose:
            print("New pattern")
            self.print_pattern()
            print("Time to next pattern: %f" % self.time_to_next_pattern)
        

    def print_pattern(self):
        cdef int i
        cdef double *pattern=<double *>self.rate.data
        for i in range(self.N):
            print(pattern[i])


    @cython.cdivision(True)
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    cpdef update(self,double t,simulation sim):
        cdef double r
        cdef int i,j
        cdef double *rate=<double *>self.rate.data
        
        cdef double *pattern
        cdef int *spiking=<int *>self.spiking.data
        
        if t>=(self.time_to_next_pattern-1e-6):  # the 1e-6 is because of binary represenation offsets
            self.new_pattern(t)
        pattern=<double *>self.rate.data    
        
        self.is_spike=0
        for i in range(self.N):
            r=randu()
            if r<(pattern[i]*sim.dt):
                self.is_spike=1
                spiking[i]=1
                if self.save_spikes_begin<=t<=self.save_spikes_end:
                    self.saved_spikes.append( (t,i) )
            else:
                spiking[i]=0


cdef class isi_pattern(poisson_pattern):

    cdef public distribution ISI
    cdef public int need_to_reset_last_spike_time

    cpdef _reset(self):
        poisson_pattern._reset(self)
        self.need_to_reset_last_spike_time=True

    def __init__(self,patterns,distribution ISI,
                    time_between_patterns=0.2,sequential=False,
                    shape=None,verbose=False):

        poisson_pattern.__init__(self,patterns,
                    time_between_patterns,sequential,shape,verbose)

        self.ISI=ISI

        s=str(ISI)
        s=s.split(' ')[0].split('.')[-1]
        self.name='ISI Pattern %s' % s

        self._reset()


    @cython.cdivision(True)
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    cpdef update(self,double t,simulation sim):
        cdef double r
        cdef int i,j
        cdef double x,cdf,pdf,_lambda
        cdef double *rate=<double *>self.rate.data
        cdef double *last_spike_time=<double *>self.last_spike_time.data
        
        cdef double *pattern
        cdef int *spiking=<int *>self.spiking.data

        if self.need_to_reset_last_spike_time:
            for i in range(self.N):
                last_spike_time[i]=-sim.dt    # assume a spike one dt ago
            self.need_to_reset_last_spike_time=False
        
        if t>=(self.time_to_next_pattern-1e-6):  # the 1e-6 is because of binary represenation offsets
            self.new_pattern(t)
        pattern=<double *>self.pattern.data    
        
        self.is_spike=0
        for i in range(self.N):

            x=t-last_spike_time[i]
            self.ISI.set_rate(pattern[i]) 

            cdf=self.ISI.cdf(x)
            pdf=self.ISI.pdf(x)
            
            if cdf==1.0:  # guarantee a spike - avoid divide by zero
                self.is_spike=1
                spiking[i]=1
            else:
                _lambda=pdf/(1-cdf)
            
                if randu()<_lambda*sim.dt:
                    self.is_spike=1
                    spiking[i]=1
                else:
                    spiking[i]=0

            if spiking[i]:
                last_spike_time[i]=t
                if self.save_spikes_begin<=t<=self.save_spikes_end:
                    self.saved_spikes.append( (t,i) )


cdef class isi_plasticnet(poisson_plasticnet):

    cdef public distribution ISI
    cdef public int need_to_reset_last_spike_time

    cpdef _reset(self):
        poisson_plasticnet._reset(self)
        self.need_to_reset_last_spike_time=True


    def __init__(self,pneuron,distribution ISI,sim=None,
                time_between_patterns=0.2,verbose=False):

        poisson_plasticnet.__init__(self,pneuron,sim=sim,
                time_between_patterns=time_between_patterns,
                verbose=verbose)

        self.ISI=ISI

        s=str(ISI)
        s=s.split(' ')[0].split('.')[-1]
        self.name='ISI Plasticnet %s' % s

        self._reset()


    @cython.cdivision(True)
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    cpdef update(self,double t,simulation sim):
        cdef double r
        cdef int i,j
        cdef double x,cdf,pdf,_lambda
        cdef double *rate=<double *>self.rate.data
        cdef double *last_spike_time=<double *>self.last_spike_time.data
        
        cdef double *pattern
        cdef int *spiking=<int *>self.spiking.data

        if self.need_to_reset_last_spike_time:
            for i in range(self.N):
                last_spike_time[i]=-sim.dt    # assume a spike one dt ago
            self.need_to_reset_last_spike_time=False
        
        if t>=(self.time_to_next_pattern-1e-6):  # the 1e-6 is because of binary represenation offsets
            self.new_pattern(t)
        pattern=<double *>self.rate.data    
        
        self.is_spike=0
        for i in range(self.N):

            x=t-last_spike_time[i]
            self.ISI.set_rate(pattern[i]) 

            cdf=self.ISI.cdf(x)
            pdf=self.ISI.pdf(x)
            
            if cdf==1.0:  # guarantee a spike - avoid divide by zero
                self.is_spike=1
                spiking[i]=1
            else:
                _lambda=pdf/(1-cdf)
            
                if randu()<_lambda*sim.dt:
                    self.is_spike=1
                    spiking[i]=1
                else:
                    spiking[i]=0

            if spiking[i]:
                last_spike_time[i]=t
                if self.save_spikes_begin<=t<=self.save_spikes_end:
                    self.saved_spikes.append( (t,i) )
