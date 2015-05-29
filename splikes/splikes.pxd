cimport numpy as np

cdef extern from "math.h":
    double sqrt(double)
    double exp(double)
    double log(double)
    double tanh(double)
    double erf(double)
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

cdef  init_by_int(int seed)
cdef  init_by_entropy()
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
    cdef public double time_to_next_save,time_to_next_filter
    cdef public object monitors,filters
    cdef public int seed
    cpdef _reset(self)

cdef class neuron(group):
    cdef public int is_spike
    cdef public int post_count    
    cdef public object saved_spikes    
    cdef public double save_spikes_begin,save_spikes_end
    cdef public np.ndarray spiking,rate
    cdef public int N
    cdef public np.ndarray last_spike_time
    cdef public connections_pre,connections_post
    cdef public int num_pre,num_post
    cdef public object state_variable
    cpdef _reset(self)
    cpdef update(self,double t,simulation sim)


cdef class connection(group):
    cdef public np.ndarray weights
    cdef public np.ndarray initial_weights
    cdef public bint reset_to_initial
    cdef public object initial_weight_range
    cdef public double w_max,w_min
    cdef public neuron pre,post
    cdef double *W
    cdef public np.ndarray state
    cdef public int use_state
    cdef public object state_variable
    cdef public double spike_scale
    cpdef _reset(self)
    cpdef update(self,double t,simulation sim)
    cpdef apply_weight_limits(self)

