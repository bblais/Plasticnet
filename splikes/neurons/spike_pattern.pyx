from splikes.splikes cimport *
cimport cython
import pylab

import numpy as np
cimport numpy as np

cdef class spike_pattern(neuron):
    cdef public np.ndarray spike_neurons
    cdef public np.ndarray spike_times
    cdef public int count
    cdef public double time_to_next_spike
    cdef public double time_between_patterns,time_to_next_pattern
    cdef public double time_begin_pattern
    cdef public int pattern_length

    cpdef _reset(self):
        neuron._reset(self)
        self.time_to_next_spike=self.spike_times[0]
        self.time_to_next_pattern=self.time_between_patterns
        self.time_begin_pattern=0.0
        self.count=0

    def __init__(self,N,spike_times,spike_neurons,time_between_patterns=0.2,verbose=False):

        neuron.__init__(self,N) # number of neurons

        self.verbose=verbose
        self.name='Spike Pattern'

        assert len(spike_times)==len(spike_neurons),"Number of spike times different than number of spike neurons"

        s=sorted(zip(spike_times,spike_neurons))
        spike_times,spike_neurons=zip(*s)

        self.spike_neurons=np.array(spike_neurons,np.int64).ravel()
        self.spike_times=np.array(spike_times,np.float).ravel()
        self.time_between_patterns=time_between_patterns


        assert self.spike_times.max()<self.time_between_patterns,'Spike times after time between patterns'
        assert self.spike_neurons.max()<N,'More spike neurons than neurons'
        assert self.spike_neurons.min()>=0,'Negative spike neurons?'
        assert self.spike_times.min()>=0.0,'Negative spike times?'

        self.pattern_length=len(self.spike_times)

        self._reset()


    def plot_spikes(self,count=False):
        cdef object x
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


    @cython.cdivision(True)
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    cpdef update(self,double t,simulation sim):
        cdef int i,j
        cdef np.int64_t *spike_neurons=<np.int64_t *>self.spike_neurons.data
        cdef double *spike_times=<double *>self.spike_times.data
        cdef int *spiking=<int *>self.spiking.data
        
        if t>=(self.time_to_next_pattern-1e-6):  # the 1e-6 is because of binary represenation offsets
            self.time_begin_pattern=t
            self.time_to_next_pattern+=self.time_between_patterns
            self.count=0
            if self.verbose:
                print("Changing patterns",t)

        for i in range(self.N):
            spiking[i]=0

        if self.count>=self.pattern_length:
            return


        while (t-self.time_begin_pattern)>=(spike_times[self.count]-1e-6):
            i=spike_neurons[self.count]

            if self.verbose:
                print("spiking ",i,"at time ",t,"spike time",spike_times[self.count],"count",self.count)

            spiking[i]=1
            self.count+=1
            if self.save_spikes_begin<=t<=self.save_spikes_end:
                self.saved_spikes.append( (t,i) )

            if self.count>=self.pattern_length:
                if self.verbose:
                    print("end of pattern.")
                break


