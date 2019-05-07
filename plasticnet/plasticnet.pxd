cimport cython

import numpy as np
cimport numpy as np

inf=1e500

from copy import deepcopy
import pylab
from Waitbar import Waitbar
from copy import deepcopy


cdef extern from "math.h":
    double sqrt(double)
    double exp(double)
    double log(double)
    double tanh(double)
    double pow(double,double)

cdef extern from "randomkit.h":
    ctypedef struct rk_state: 
        pass
    ctypedef struct rk_error: 
        pass
    
    void rk_seed(unsigned long seed, rk_state *state)
    rk_error rk_randomseed(rk_state *state)
    double rk_double(rk_state *state)
    double rk_gauss(rk_state *state)

cpdef  init_by_int(int seed)
cpdef  init_by_entropy()
cdef double randu()
cdef double randn()
cdef double rande()
 
cdef class group:
    cdef public object save_attrs,save_data
    cdef public object name
    cdef public int verbose


cdef class monitor(group):
    cdef public double time_to_next_save
    cdef public double save_interval
    cdef public object container
    cdef public object t,values
    cdef object t_tmp,values_tmp
    cpdef update(self,double t)

cdef class simulation(group):
    cdef public double dt
    cdef public double total_time
    cdef public double start_time
    cdef public double current_time
    cdef public double time_to_next_save,time_to_next_filter
    cdef public object monitors,filters
    cdef public post_process
    cdef public int seed

    cpdef _reset(self)

cdef class neuron(group):
    cdef public int N
    cdef public np.ndarray output,linear_output
    cdef public connections_pre,connections_post
    cdef public post_process
    cdef public int num_pre,num_post
    cpdef _reset(self)
    cpdef update(self,double t,simulation sim)
    cpdef _clean(self)

cdef class post_process_neuron(group):
    cpdef _reset(self)
    cpdef update(self,double t,simulation sim)
    cdef public neuron n

cdef class post_process_channel(group):
    cdef public channel ch
    cpdef _reset(self)
    cpdef update(self,double t,simulation sim)

cdef class channel(neuron):
    cdef public object neuron_list
    cdef public int number_of_neurons
    cdef public double time_between_patterns,time_to_next_pattern
 
cdef class connection(group):
    cdef public np.ndarray weights
    cdef public np.ndarray initial_weights
    cdef public object initial_weight_range
    cdef public neuron pre,post
    cdef double *w
    cdef public bint reset_to_initial
    cdef public post_process

    cpdef _reset(self)
    cpdef update(self,double t,simulation sim)
    cpdef _clean(self)

cdef class post_process_connection(group):
    cpdef _reset(self)
    cpdef update(self,double t,simulation sim)
    cdef public connection c
    
