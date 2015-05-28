
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')
from splikes import *


# ## 1D spiking neuron

# In[2]:

patterns=array([10])
pre=neurons.poisson_pattern(patterns)
pre.time_between_patterns=2
pre.save_spikes_begin=0.0
pre.save_spikes_end=10.0

sim=simulation(10,dt=0.0001)


# In[3]:

get_ipython().magic(u'time run_sim(sim,[pre],[])')
pre.plot_spikes()


# spike counts per second

# In[4]:

spike_counts(arange(0,10+1),pre.saved_spikes)


# ### 1D Non-constant rates

# In[12]:

pre=neurons.poisson_pattern([5,50],
                            sequential=True,
                            )
pre.time_between_patterns=2
pre.save_spikes_begin=0.0
pre.save_spikes_end=10.0

sim=simulation(10,dt=0.0001)

get_ipython().magic(u'time run_sim(sim,[pre],[])')
pre.plot_spikes()
title('Oops!  This is two neurons!')

print spike_counts(arange(0,10+1),pre.saved_spikes)


# In[13]:

pre=neurons.poisson_pattern([5,50],
                            shape=(2,1),
                            sequential=True,
                            )
pre.time_between_patterns=2
pre.save_spikes_begin=0.0
pre.save_spikes_end=10.0

sim=simulation(10,dt=0.0001)

get_ipython().magic(u'time run_sim(sim,[pre],[])')
pre.plot_spikes()


# ## 1D SRM0 neuron

# In[12]:

pre=neurons.poisson_pattern([10])
post=neurons.srm0(1)

c=connection(pre,post,[1,1])

sim=simulation(10,dt=0.0001)
sim.monitor(post,['u',],0.001)

run_sim(sim,[pre,post],[c])


# In[13]:

sim.monitors['u'].array()


# In[14]:

m=sim.monitors['u']
m.plot()


# ### Checking the effect connection strength

# In[16]:

pre=neurons.poisson_pattern([10])
post=neurons.srm0(1)

c=connection(pre,post,[10,10])

sim=simulation(10,dt=0.0001)
sim.monitor(post,['u',],0.001)

run_sim(sim,[pre,post],[c])

m=sim.monitors['u']
m.plot()


# In[17]:

mean(m.array())


# ### Running many different connection strengths

# In[22]:

w_arr=linspace(1,100,100)
print w_arr


# In[24]:

mean_arr=[]
rate=10
for w in w_arr:
    
    pre=neurons.poisson_pattern([rate])
    post=neurons.srm0(1)
    
    c=connection(pre,post,[w,w])
    
    sim=simulation(10,dt=0.0001)
    sim.monitor(post,['u',],0.001)
    
    run_sim(sim,[pre,post],[c],print_time=False)
    
    u=sim.monitors['u'].array()
    mean_arr.append(mean(u))
    
plot(w_arr,mean_arr,'o')
xlabel('Connection Strength')
ylabel('Mean $u$')
title('Input Rate %.1f' % rate)


# In[25]:

mean_arr=[]
rate=30
for w in w_arr:
    
    pre=neurons.poisson_pattern([rate])
    post=neurons.srm0(1)
    
    c=connection(pre,post,[w,w])
    
    sim=simulation(10,dt=0.0001)
    sim.monitor(post,['u',],0.001)
    
    run_sim(sim,[pre,post],[c],print_time=False)
    
    u=sim.monitors['u'].array()
    mean_arr.append(mean(u))
    
plot(w_arr,mean_arr,'o')
xlabel('Connection Strength')
ylabel('Mean $u$')
title('Input Rate %.1f' % rate)


# Can you figure out an equation for the mean $u$ for a given connection strength and input rate?

# ## 2D Spiking Neuron

# In[15]:

pre=neurons.poisson_pattern([[10,20],[50,10]],
                            sequential=True,
                            verbose=True
                            )
pre.time_between_patterns=2
pre.save_spikes_begin=0.0
pre.save_spikes_end=10.0


post=neurons.srm0(1)

c=connection(pre,post,[1,1])

sim=simulation(10,dt=0.0001)
sim.monitor(post,['u',],0.001)

run_sim(sim,[pre,post],[c])


figure()
pre.plot_spikes()


figure()
m=sim.monitors['u']
m.plot()


# Can you figure out an equation to provide the mean $u$ given the two input rate values and the two connection strengths?

# In[ ]:



