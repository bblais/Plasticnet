#!/usr/bin/env python
# coding: utf-8

# In[1]:


from splikes import *


# In[13]:


from pylab import figure,legend,plot,linspace


# <img src="./images/epsp1.png">

# Input Rate: $\rho$
# 
# Equation for Unsmoothed:
# 
# \begin{eqnarray}
# u&\rightarrow& u+a\cdot w_i \mbox{ (input spike on input $i$)}\\
# \frac{du}{dt}&=& -u/\tau
# \end{eqnarray}
# 
# Average value:
# \begin{eqnarray}
# <u> = a\cdot w\cdot \tau\cdot \rho
# \end{eqnarray}
# 
# Equation for Smoothed:
# 
# \begin{eqnarray}
# v&\rightarrow& v+a\cdot w_i \mbox{ (input spike on input $i$)}\\
# \frac{dv}{dt}&=& -v/\tau\\
# \frac{du}{dt}&=& (v-u)/\tau
# \end{eqnarray}
# 
# Average value:
# \begin{eqnarray}
# <u> = a\cdot w\cdot \tau\cdot \rho
# \end{eqnarray}
# 

# In[4]:


pre=neurons.poisson_pattern([10])

post=neurons.srm0(1)
post.smoothed=True
post.tau=0.01

post2=neurons.srm0(1)
post2.smoothed=False
post2.tau=0.01
post2.name='unsmoothed'

c=connection(pre,post,[1,1])
c2=connection(pre,post2,[1,1])

sim=simulation(.3,dt=0.0001)
sim.monitor(post,['u',],0.001)
sim.monitor(post2,['u',],0.001)

run_sim(sim,[pre,post,post2],[c,c2])

figure(figsize=(10,3))
m=sim.monitors['u']
m.plot()
m=sim.monitors['u [unsmoothed]']
m.plot()
legend(['Smoothed','Unsmoothed'])


# In[6]:


pre=neurons.poisson_pattern([20])
pre.save_spikes_begin=0.0
pre.save_spikes_end=10

post=neurons.srm0(1)
post.smoothed=True
post.tau=0.01

c=connection(pre,post,[1,1])

sim=simulation(2,dt=0.0001)
sim.monitor(post,['u',],0.001)

run_sim(sim,[pre,post],[c])

figure(figsize=(15,5))
m=sim.monitors['u']
m.plot()

for t,n in pre.saved_spikes:
    plot([t,t],[0,0.1],'g',linewidth=3)


# In[7]:


pre.saved_spikes


# In[8]:


pre=neurons.poisson_pattern([10])

post=neurons.srm0(1)
post.smoothed=True
post.tau=0.1
post.a=10

post2=neurons.srm0(1)
post2.smoothed=False
post2.tau=0.1
post2.a=10
post2.name='unsmoothed'

c=connection(pre,post,[1,1])
c2=connection(pre,post2,[1,1])

sim=simulation(10,dt=0.0001)
sim.monitor(post,['u',],0.001)
sim.monitor(post2,['u',],0.001)

run_sim(sim,[pre,post,post2],[c,c2])

figure(figsize=(10,5))
m=sim.monitors['u']
m.plot()
m=sim.monitors['u [unsmoothed]']
m.plot()
legend(['Smoothed','Unsmoothed'])

plot([0,11],[10,10],'r--',linewidth=3)

paramtext(0.15,0.7,
          r'%d Hz' % (10),
          r'$a=%.f$' % (post2.a),
          r'$\tau=%.1f$'  % (post2.tau),
          )


# ### try with isi invgauss input

# In[9]:


ISI=neurons.isi_distributions.invgauss(0,1.0)
pre=neurons.isi_pattern([10],ISI)
pre.time_between_patterns=1*second
pre.save_spikes_begin=0
pre.save_spikes_end=10

post=neurons.srm0(1)
post.smoothed=True
post.tau=0.1
post.a=10
post.save_spikes_begin=0
post.save_spikes_end=10

post2=neurons.srm0(1)
post2.smoothed=False
post2.tau=0.1
post2.a=10
post2.name='unsmoothed'
post2.save_spikes_begin=0
post2.save_spikes_end=10

c=connection(pre,post,[1,1])
c2=connection(pre,post2,[1,1])

sim=simulation(10,dt=0.0001)
sim.monitor(post,['u',],0.001)
sim.monitor(post2,['u',],0.001)

run_sim(sim,[pre,post,post2],[c,c2])

figure(figsize=(10,5))
m=sim.monitors['u']
m.plot()
m=sim.monitors['u [unsmoothed]']
m.plot()
legend(['Smoothed','Unsmoothed'])

plot([0,11],[10,10],'r--',linewidth=3)

paramtext(0.15,0.7,
          r'%d Hz' % (10),
          r'$a=%.f$' % (post2.a),
          r'$\tau=%.1f$'  % (post2.tau),
          )

figure()
pre.plot_spikes()

figure()
post.plot_spikes()
post2.plot_spikes(1)


# In[10]:


ISI=neurons.isi_distributions.invgauss(0,1.0)
pre=neurons.isi_pattern([10],ISI)
pre.time_between_patterns=1*second
pre.save_spikes_begin=0
pre.save_spikes_end=10

ISI2a=neurons.isi_distributions.invgauss(0,1.0)
ISI2b=neurons.isi_distributions.invgauss(0,1.0)

post=neurons.srm0_isi(1,ISI2a)
post.smoothed=True
post.tau=0.1
post.a=10
post.save_spikes_begin=0
post.save_spikes_end=10

post2=neurons.srm0_isi(1,ISI2b)
post2.smoothed=False
post2.tau=0.1
post2.a=10
post2.name='unsmoothed'
post2.save_spikes_begin=0
post2.save_spikes_end=10

c=connection(pre,post,[1,1])
c2=connection(pre,post2,[1,1])

sim=simulation(10,dt=0.0001)
sim.monitor(post,['u',],0.001)
sim.monitor(post2,['u',],0.001)

run_sim(sim,[pre,post,post2],[c,c2])

figure(figsize=(10,5))
m=sim.monitors['u']
m.plot()
m=sim.monitors['u [unsmoothed]']
m.plot()
legend(['Smoothed','Unsmoothed'])

plot([0,11],[10,10],'r--',linewidth=3)

paramtext(0.15,0.7,
          r'%d Hz' % (10),
          r'$a=%.f$' % (post2.a),
          r'$\tau=%.1f$'  % (post2.tau),
          )

figure()
pre.plot_spikes()

figure()
post.plot_spikes()
post2.plot_spikes(1)


# In[11]:


c.weights


# <img src="images/input_rate1.png">

# In[15]:


from pylab import mean


# In[16]:


rate_arr=linspace(1,50,100)
#print rate_arr

mean_arr=[]
for rate in rate_arr:
    
    pre=neurons.poisson_pattern([rate])
    post=neurons.srm0(1)
    post.tau=0.1
    post.a=10.0
    
    c=connection(pre,post,[1,1])
    
    sim=simulation(10,dt=0.0001)
    sim.monitor(post,['u',],0.001)
    
    run_sim(sim,[pre,post],[c],print_time=False)
    
    u=sim.monitors['u'].array()
    mean_arr.append(mean(u))
    
plot(rate_arr,mean_arr,'o')
xlabel(r'Input Rate ($\rho$)')
ylabel('Mean $u$')

plot(rate_arr,rate_arr*post.a*post.tau,'r--')

paramtext(.2,.7,
          r'$a=%s$' % post.a,          
          r'$\tau=%s$' % post.tau,
          r'$w=%s$' % float(c.weights),
          )

paramtext(.5,.9,
          r'$\langle u \rangle = w\cdot \rho \cdot a \cdot \tau$')


# <img src="images/weight_dependence1.png">

# In[17]:


w_arr=linspace(0.01,2,100)
#print w_arr

mean_arr=[]
rate=10
for w in w_arr:
    
    pre=neurons.poisson_pattern([rate])
    post=neurons.srm0(1)
    post.tau=0.1
    post.a=10.0
    
    c=connection(pre,post,[w,w])
    
    sim=simulation(10,dt=0.0001)
    sim.monitor(post,['u',],0.001)
    
    run_sim(sim,[pre,post],[c],print_time=False)
    
    u=sim.monitors['u'].array()
    mean_arr.append(mean(u))
    
plot(w_arr,mean_arr,'o')
xlabel('Connection Strength')
ylabel('Mean $u$')

plot(w_arr,w_arr*rate*post.a*post.tau,'r--')

paramtext(.2,.7,
          r'$a=%s$' % post.a,          
          r'$\tau=%s$' % post.tau,
          r'$\rho=%s$' % rate,
          )

paramtext(.5,.9,
          r'$\langle u \rangle = w\cdot \rho \cdot a \cdot \tau$')


# In[ ]:




