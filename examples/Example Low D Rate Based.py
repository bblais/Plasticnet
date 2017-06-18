
# coding: utf-8

# In[6]:

get_ipython().magic('pylab inline')


# In[7]:

from plasticnet import *


# ## 1D BCM

# In[8]:

from plasticnet import *
pre=neurons.pattern_neuron([10])
post=neurons.linear_neuron(1)

c=connections.BCM(pre,post,[0,.05])
c.eta=5e-7
c.tau=1000

sim=simulation(1000*1000)
sim.monitor(c,['weights','theta'],1000)

run_sim(sim,[pre,post],[c],display_hash=False)


# In[9]:

w=sim.monitors['weights'].array().squeeze()
plot(w)
xlabel('Weights')
ylabel('Time')


# ## 2D BCM

# In[10]:

pre=neurons.pattern_neuron([[2,3],[3,1]])
post=neurons.linear_neuron(1)

c=connections.BCM(pre,post,[0,.05])
c.eta=5e-5
c.tau=1000

sim=simulation(1000*1000)
sim.monitor(c,['weights','theta'],1000)

run_sim(sim,[pre,post],[c],display_hash=False)


# In[11]:

weights=sim.monitors['weights'].array().squeeze()
plot(weights)
legend(['Weight 0','Weight 1'])
ylabel('Weights')
xlabel('Time')

figure()
theta=sim.monitors['theta'].array().squeeze()
plot(theta)
ylabel(r'$\theta_M$')
xlabel('Time')


# In[12]:

outputs=[]
for w in weights:
    output=[sum(x*w) for x in pre.patterns]
    outputs.append(output)
outputs=array(outputs)

plot(outputs)
xlabel('Time')
ylabel('Response')
legend(['Pattern 0','Pattern 1'])


# ## 2D Hebb

# In[13]:

pre=neurons.pattern_neuron([[2,3],[3,1]])
post=neurons.linear_neuron(1)

c=connections.Hebb(pre,post,[0,.05])
c+=connections.process.normalization()

c.eta=5e-5
c.tau=1000

sim=simulation(1000*1000)
sim.monitor(c,['weights','theta'],1000)

run_sim(sim,[pre,post],[c],display_hash=False)


# In[14]:

weights=sim.monitors['weights'].array().squeeze()
plot(weights)
legend(['Weight 0','Weight 1'])
ylabel('Weights')
xlabel('Time')

figure()
theta=sim.monitors['theta'].array().squeeze()
plot(theta)
ylabel(r'$\theta_M$')
xlabel('Time')


# In[15]:

outputs=[]
for w in weights:
    output=[sum(x*w) for x in pre.patterns]
    outputs.append(output)
outputs=array(outputs)

plot(outputs)
xlabel('Time')
ylabel('Response')
legend(['Pattern 0','Pattern 1'])


# In[ ]:



