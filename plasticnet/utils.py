import matplotlib as mpl
import pylab as pl

import h5py
import plasticnet as pn
import splikes as sp

def save(fname,sim,neurons=[],connections=[]):
    f=h5py.File(fname,'w')
    
    try:

        f.attrs['plasticnet version']=pn.version
        f.attrs['splikes version']=sp.version
        
        group=f.create_group("simulation")
        sim.save(group)

        for n,neuron in enumerate(neurons):
            group=f.create_group("neuron %d" % n)
            if neuron.verbose:
                print "<<<<  group   neuron %d >>>>" % n
                sys.stdout.flush()
            neuron.save(group)

            for monitor_name in sim.monitors:
                m=sim.monitors[monitor_name]
                if m.container==neuron:
                    mgroup=group.create_group("monitor %s" % m.name)
                    m.save(mgroup)
            
            
            
        for c,connection in enumerate(connections):
            group=f.create_group("connection %d" % c)
            
            if connection.verbose:
                print "<<<<  group   connection %d >>>>" % c
                sys.stdout.flush()
            connection.save(group)
    
            try:
                idx=neurons.index(connection.pre)
            except ValueError:
                idx=None
            group.attrs['pre number']=idx

            try:
                idx=neurons.index(connection.post)
            except ValueError:
                idx=None
            group.attrs['post number']=idx
            
            for monitor_name in sim.monitors:
                m=sim.monitors[monitor_name]
                if m.container==connection:
                    mgroup=group.create_group("monitor %s" % m.name)
                    m.save(mgroup)
            
    finally:
        f.close()





def hdf5_load_images(fname):
    import h5py,os
    import numpy as np
    
    if not os.path.exists(fname):
        raise ValueError,"File does not exist: %s" % fname
    f=h5py.File(fname,'r')
    var={}
    var['im_scale_shift']=list(f.attrs['im_scale_shift'])
    N=len(f.keys())
    var['im']=[]
    for i in range(N):
        var['im'].append(np.array(f['image%d' % i]))

    f.close()

    return var
    
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['figure.figsize'] = (10,8)
mpl.rcParams['axes.grid']=True

def bigfonts(size=20,family='sans-serif'):

    from matplotlib import rc

    rc('font',size=size,family=family)
    rc('axes',labelsize=size)
    rc('axes',titlesize=size)
    rc('xtick',labelsize=size)
    rc('ytick',labelsize=size)
    rc('legend',fontsize=size)

bigfonts()

def time2str(t):

    minutes=60
    hours=60*minutes
    days=24*hours
    years=365*days
    
    yr=int(t/years)
    t-=yr*years

    dy=int(t/days)
    t-=dy*days
    
    hr=int(t/hours)
    t-=hr*hours

    mn=int(t/minutes)
    t-=mn*minutes

    sec=t

    s=""
    if yr>0:
        s+=str(yr)+" years "
    if hr>0:
        s+=str(hr)+" hours "
    if mn>0:
        s+=str(mn)+" minutes "        
        
    s+=str(sec)+" seconds "


    return s

def timeit(reset=False):
    from time import time
    
    global _timeit_time
    
    try:
        if _timeit_time is None:
            pass
        # is defined
    except NameError:
        _timeit_time=time()
        print "Time Reset"
        return
    
    if reset:
        _timeit_time=time()
        print "Time Reset"
        return

    return time2str(time()-_timeit_time)

def test_stim(sim,neurons,connections,numang=24,k=4.4/13.0*3.141592653589793235):
    from numpy import cos,sin,arctan2,linspace,mgrid,pi

    pre,post=neurons
    c=connections[0]

    try:
        rf_diameter=pre.rf_size
    except AttributeError:
        rf_diameter=pre.neuron_list[0].rf_size

    theta=linspace(0.0,pi,numang)+pi/2
    x=linspace(0.0,pi,numang)*180.0/pi
    
    i,j= mgrid[-rf_diameter//2:rf_diameter//2,
                    -rf_diameter//2:rf_diameter//2]
    i=i+1
    j=j+1
    
    i=i.ravel()
    j=j.ravel()
    
    sine_gratings=[]
    cosine_gratings=[]
    
    for t in theta:
        kx=k*cos(t)
        ky=k*sin(t)
        
        
        sine_gratings.append(sin(kx*i+ky*j))   # sin grating input (small amp)
        cosine_gratings.append(cos(kx*i+ky*j))   # cos grating input (small amp)


    m=sim.monitors['weights']
    time_mat=m.saved_results['t']
    weights_mat=m.saved_results['weights']
    
    num_neurons=len(weights_mat[0])


    results=[]

    for t,weights in zip(time_mat,weights_mat): #loop over time
        for i,w in enumerate(weights):  #loop over neurons

            try:
                rf_size=pre.rf_size
                neurons=[pre]
            except AttributeError:
                neurons=pre.neuron_list

            num_channels=len(neurons)

            one_result=[]
            count=0
            for c,ch in enumerate(neurons):   
                rf_size=ch.rf_size
                N=ch.N
                weights_1channel_1neuron=w[count:(count+rf_size*rf_size)]


                y=[]
                for ds,dc in zip(sine_gratings,cosine_gratings):
                
                    cs=(weights_1channel_1neuron*ds).sum() # response to sin/cos grating input
                    cc=(weights_1channel_1neuron*dc).sum()
                    
                    phi=arctan2(cc,cs)  # phase to give max response
                    
                    c=cs*cos(phi)+cc*sin(phi)     # max response
                
                    y.append(c)
                    
                val=(t,max(y),x,y)
          
                one_result.append(val)

                count+=rf_size*rf_size
            results.append(one_result)
            
    return results

def plot_test_stim(results):
    from pylab import subplot,plot
    from numpy import array

    subplot(1,2,1)
    result=results[-1]
    for r in result:
        t,mx,x,y=r
        plot(x,y,'-o')

    t_mat=[]
    y_mat=[]
    for result in results:
        t_mat.append(result[0][0])
        yy=[]
        for r in result:
            t,mx,x,y=r
            yy.append(mx)
        y_mat.append(yy)

    t_mat=array(t_mat)
    y_mat=array(y_mat)

    subplot(1,2,2)
    plot(t_mat,y_mat,'-')


def plot_rfs_and_theta(sim,neurons,connections):

    pre,post=neurons
    c=connections[0]
    
    weights=c.weights
    
    num_neurons=len(weights)
    fig=pl.figure(figsize=(16,4*num_neurons))
    for i,w in enumerate(weights):
        try:
            rf_size=pre.rf_size
            neurons=[pre]
        except AttributeError:
            neurons=pre.neuron_list

        num_channels=len(neurons)

        count=0
        vmin,vmax=w.min(),w.max()
        for c,ch in enumerate(neurons):   
            rf_size=ch.rf_size
            pl.subplot2grid((num_neurons,num_channels+1),(i, c),aspect='equal')
            subw=w[count:(count+rf_size*rf_size)]
            #pl.pcolor(subw.reshape((rf_size,rf_size)),cmap=pl.cm.gray)
            pl.pcolormesh(subw.reshape((rf_size,rf_size)),cmap=pl.cm.gray,
                vmin=vmin,vmax=vmax)
            pl.xlim([0,rf_size]); 
            pl.ylim([0,rf_size])
            pl.axis('off')
            count+=rf_size*rf_size

        pl.subplot2grid((num_neurons,num_channels+1),(i, num_channels))
        sim.monitors['theta'].plot()

    return fig

def plot_rfs(sim,neurons,connections):

    pre,post=neurons
    c=connections[0]
    
    weights=c.weights
    
    num_neurons=len(weights)
    fig=pl.figure(figsize=(16,4*num_neurons))
    for i,w in enumerate(weights):
        try:
            rf_size=pre.rf_size
            neurons=[pre]
        except AttributeError:
            neurons=pre.neuron_list

        num_channels=len(neurons)

        count=0
        vmin,vmax=w.min(),w.max()
        for c,ch in enumerate(neurons):   
            rf_size=ch.rf_size
            pl.subplot2grid((num_neurons,num_channels),(i, c),aspect='equal')
            subw=w[count:(count+rf_size*rf_size)]
            #pl.pcolor(subw.reshape((rf_size,rf_size)),cmap=pl.cm.gray)
            pl.pcolormesh(subw.reshape((rf_size,rf_size)),cmap=pl.cm.gray,
                vmin=vmin,vmax=vmax)
            pl.xlim([0,rf_size]); 
            pl.ylim([0,rf_size])
            pl.axis('off')
            count+=rf_size*rf_size

    return fig

def get_output_distributions(neurons,total_time=100000,display_hash=False):
    import plasticnet as pn
    total_time=10000
    sim=pn.simulation(total_time)
    
    for neuron in neurons:
        sim.monitor(neuron,['output'],1)
        
    pn.run_sim(sim,neurons,[],display_hash=display_hash)
    
    outs=[]
    for key in sim.monitors:
        m=sim.monitors[key]
        t,out=m.arrays()
        outs.append(out)
        
    return outs

def plot_output_distribution(out,title):
    from splikes.utils import paramtext

    out=out.ravel()
    out_full=out

    result=pl.hist(out,200)
    paramtext(1.2,0.95,
              'min %f' % min(out_full),
              'max %f' % max(out_full),
              'mean %f' % pl.mean(out_full),
              'median %f' % pl.median(out_full),
              'std %f' % pl.std(out_full),
              )
    pl.title(title)
    
    