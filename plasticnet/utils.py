import matplotlib as mpl
import pylab as py

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import h5py

second=1
ms=0.001*second
minute=60*second
hour=60*minute
day=24*hour


import plasticnet as pn
import splikes as sp

import asdf
import warnings
warnings.filterwarnings("ignore",category=asdf.exceptions.AsdfDeprecationWarning)

def asdf_load_images(fname):
    var={}
    with asdf.open(fname) as af:
        var['im_scale_shift']=af.tree['im_scale_shift']
        var['im']=[array(_) for _ in af.tree['im']]

    return var


class Tree(dict):

    def __init__(self, *args):
        dict.__init__(self, args)
        self['attrs']={}
        self.attrs=self['attrs']

    def create_group(self,name):
        self[name]=Tree()
        return self[name]

    def create_dataset(self,name,data):
        self[name]=data


    def todict(self):
        for key in self:
            if isinstance(self[key],Tree):
                self[key]=self[key].todict()
        return(dict(self))

class Sequence(object):
    
    def __init__(self):
        self.sims=[]
        self.neurons=[]
        self.connections=[]
        self.length=0
        self.save_attrs=['length']
        self.save_data=[]

    def __len__(self):
        return len(self.sims)
    
    def __iadd__(self,other):
        s,n,c=other
        self.sims.append(s)
        self.neurons.append(n)
        self.connections.append(c)
        self.length=len(self.sims)
        
        return self
    
    def run(self,**kwargs):

        for i in range(len(self.sims)):
            s,ns,cs=self.sims[i],self.neurons[i],self.connections[i]
            
            if i>0: # load from previous
                s0,ns0,cs0=self.sims[i-1],self.neurons[i-1],self.connections[i-1]
                
                for c,c0 in zip(cs,cs0):
                    c.initial_weights=c0.weights.copy()
                    c.reset_to_initial=True

                    try:
                        c.initial_theta=c0.theta.copy()
                    except AttributeError:
                        pass

                    c._reset()
                
                start_time=s0.start_time+s0.total_time+s0.dt
                s.start_time=start_time
                s.current_time=start_time

                for key in s.monitors:
                    s.monitors[key].time_to_next_save=start_time
                
            pn.run_sim(s,ns,cs,**kwargs)

    def arrays(self,name):
        return self.time_array(name),self.array(name)
        
    def array(self,name):
        import numpy as np
        from itertools import chain
        v=[s.monitors[name].values for s in self.sims]
        v=list(chain(*v))        
        return np.array(v).squeeze()
        
    def time_array(self,name):
        import numpy as np
        from itertools import chain
        v=[s.monitors[name].t for s in self.sims]
        v=list(chain(*v))        
        return np.array(v)
            
    def plot(self,name):
        for i in range(len(self.sims)):
            s,ns,cs=self.sims[i],self.neurons[i],self.connections[i]
            s.monitors[name].plot()
            
    def __getitem__(self, index):
        result = self.sims[index],self.neurons[index],self.connections[index]
        return result
            

def saveold(fname,seq,neurons=[],connections=[]):
    import sys
    assert '.asdf' in fname

    f=Tree()
    
    f.attrs['plasticnet version']=pn.version
    f.attrs['splikes version']=sp.version
    
    try:
        first_sim=seq[0] 
        # sequence
    except TypeError:
        sim=seq
        seq=Sequence()
        seq+=seq,neurons,connections

    seqgroup=f.create_group("sequence")

    group=f.create_group("simulation")
    sim.save(group)

    for n,neuron in enumerate(neurons):
        group=f.create_group("neuron_%d" % n)
        if neuron.verbose:
            print("<<<<  group   neuron %d >>>>" % n)
            sys.stdout.flush()
        neuron.save(group)

        for monitor_name in sim.monitors:
            m=sim.monitors[monitor_name]
            if m.container==neuron:
                mgroup=group.create_group("monitor_%s" % m.name)
                m.save(mgroup)
        
        
        
    for c,connection in enumerate(connections):
        group=f.create_group("connection %d" % c)
        
        if connection.verbose:
            print("<<<<  group   connection_%d >>>>" % c)
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
                mgroup=group.create_group("monitor_%s" % m.name)
                m.save(mgroup)
        
    if sim.verbose:
        print("Saving %s" % fname)

    af = asdf.AsdfFile(f.todict())
    af.write_to(fname, all_array_compression='zlib')


def save(fname,seq,neurons=[],connections=[]):
    import sys,os
    assert '.asdf' in fname

    top=Tree()
    
    top.attrs['plasticnet version']=pn.version
    top.attrs['splikes version']=sp.version
    
    if not isinstance(seq,Sequence):
        sim=seq
        seq=Sequence()
        seq+=sim,neurons,connections

    top.attrs['sequence length']=len(seq)

    for ns, (sim, neurons, connections) in enumerate(seq):
        f=top.create_group("sequence %d" % ns)

        group=f.create_group("simulation")
        sim.save(group)

        for n,neuron in enumerate(neurons):
            group=f.create_group("neuron %d" % n)
            if neuron.verbose:
                print("<<<<  group   neuron %d >>>>" % n)
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
                print("<<<<  group   connection %d >>>>" % c)
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
            
    if sim.verbose:
        print("Saving %s" % fname)

    af = asdf.AsdfFile(top.todict())

    base,name=os.path.split(fname)
    if not(os.path.exists(base)):
        print("Making %s" % base)
        os.makedirs(base)


    af.write_to(fname, all_array_compression='zlib')


def hdf5_load_images(fname):
    import h5py,os
    import numpy as np
    
    if not os.path.exists(fname):
        raise ValueError("File does not exist: %s" % fname)
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


def timeit(reset=False):
    from time import time
    
    global _timeit_time
    
    try:
        if _timeit_time is None:
            pass
        # is defined
    except NameError:
        _timeit_time=time()
        #print("Time Reset")
        return
    
    if reset:
        _timeit_time=time()
        #print("Time Reset")
        return

    return time2str(time()-_timeit_time)

def test_stim(sim,neurons,connections,numang=24,k=4.4/13.0*3.141592653589793235):
    from numpy import cos,sin,arctan2,linspace,mgrid,pi

    pre,post=neurons
    c=connections[0]

    try:  # check for channel
        neurons=pre.neuron_list
    except AttributeError:
        neurons=[pre]

    num_channels=len(neurons)

    ## only works for all the same size, for right now
    all_sizes=[]
    for c,ch in enumerate(neurons):   
        try:
            rf_size=ch.rf_size
        except AttributeError:
            rf_size=py.sqrt(ch.N)
            assert rf_size==int(rf_size)
            rf_size=int(rf_size)

        all_sizes.append(rf_size)

    assert all([x==all_sizes[0] for x in all_sizes])

    rf_diameter=rf_size

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
    time_mat=m.t
    weights_mat=m.values
    
    num_neurons=len(weights_mat[0])


    results=[]

    for t,weights in zip(time_mat,weights_mat): #loop over time
        for i,w in enumerate(weights):  #loop over neurons

            one_result=[]
            count=0
            for c,ch in enumerate(neurons):   
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
    fig=py.figure(figsize=(16,4*num_neurons))

    for i,w in enumerate(weights):
        try:  # check for channel
            neurons=pre.neuron_list
        except AttributeError:
            neurons=[pre]

        num_channels=len(neurons)

        count=0
        vmin,vmax=w.min(),w.max()
        for c,ch in enumerate(neurons):   
            try:
                rf_size=ch.rf_size
                if rf_size<0:
                    rf_size=py.sqrt(ch.N)
                    assert rf_size==int(rf_size)
                    rf_size=int(rf_size)

            except AttributeError:
                rf_size=py.sqrt(ch.N)
                assert rf_size==int(rf_size)
                rf_size=int(rf_size)


            py.subplot2grid((num_neurons,num_channels+1),(i, c),aspect='equal')
            subw=w[count:(count+rf_size*rf_size)]
            #py.pcolor(subw.reshape((rf_size,rf_size)),cmap=py.cm.gray)
            py.pcolormesh(subw.reshape((rf_size,rf_size)),cmap=py.cm.gray,
                vmin=vmin,vmax=vmax)
            py.xlim([0,rf_size]); 
            py.ylim([0,rf_size])
            py.axis('off')
            count+=rf_size*rf_size

    py.subplot2grid((num_neurons,num_channels+1),(0, num_channels))
    sim.monitors['theta'].plot()

    return fig

def plot_rfs(sim,neurons,connections):

    pre,post=neurons
    c=connections[0]
    
    weights=c.weights
    
    num_neurons=len(weights)

    try:  # check for channel
        neurons=pre.neuron_list
    except AttributeError:
        neurons=[pre]

    num_channels=len(neurons)



    fig=py.figure(figsize=(16,4*num_neurons))
    for i,w in enumerate(weights):

        count=0
        vmin,vmax=w.min(),w.max()
        for c,ch in enumerate(neurons):   

            try:
                rf_size=ch.rf_size
            except AttributeError:
                rf_size=py.sqrt(ch.N)
                assert rf_size==int(rf_size)
                rf_size=int(rf_size)


            rf_size=ch.rf_size
            py.subplot2grid((num_neurons,num_channels),(i, c),aspect='equal')
            subw=w[count:(count+rf_size*rf_size)]
            #py.pcolor(subw.reshape((rf_size,rf_size)),cmap=py.cm.gray)
            py.pcolormesh(subw.reshape((rf_size,rf_size)),cmap=py.cm.gray,
                vmin=vmin,vmax=vmax)
            py.xlim([0,rf_size]); 
            py.ylim([0,rf_size])
            py.axis('off')
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

    result=py.hist(out,200)
    paramtext(1.2,0.95,
              'min %f' % min(out_full),
              'max %f' % max(out_full),
              'mean %f' % py.mean(out_full),
              'median %f' % py.median(out_full),
              'std %f' % py.std(out_full),
              )
    py.title(title)
    
from numba import njit,jit
import numpy
from math import sin,cos,atan2


@njit
def get_gratings(rf_diameter,theta,k_mat):

    numang=len(theta)
    num_k=len(k_mat)
    rf_area=rf_diameter*rf_diameter
    rf_radius=rf_diameter//2
    
    gratings=numpy.zeros((num_k,numang,rf_diameter,rf_diameter))

    for ki,k in enumerate(k_mat):
        for ai,th in enumerate(theta):
            a=3.14159265-th/180.0*3.14159265
            kx=k*cos(a)
            ky=k*sin(a)

            for i in range(rf_diameter):
                for j in range(rf_diameter):

                    ds=sin(kx*(i-rf_radius)+ ky*(j-rf_radius))
                    dc=cos(kx*(i-rf_radius)+ ky*(j-rf_radius))

                    gratings[ki,ai,i,j]=ds
                   
                
    return gratings

    
    
@njit
def get_responses(t,w,number_of_channels,rf_diameter,theta,k_mat):

    # weights.shape must be = (501, 3, 338)  t, neurons, inputs
    
    numang=len(theta)
    num_k=len(k_mat)
    L=len(t)
    num_neurons=w.shape[1]
    
    responses=numpy.zeros((num_k,numang,number_of_channels,num_neurons,L))

    rf_area=rf_diameter*rf_diameter
    rf_radius=rf_diameter//2
    
    for ti in range(L):
        for ni in range(num_neurons):
            for ch in range(number_of_channels):
                for ki,k in enumerate(k_mat):
                    for ai,th in enumerate(theta):
                        a=3.14159265-th/180.0*3.14159265
                        kx=k*cos(a)
                        ky=k*sin(a)

                        cs=0.0
                        cc=0.0
                        wi=0
                        for i in range(rf_diameter):
                            for j in range(rf_diameter):

                                ds=sin(kx*(i-rf_radius)+ ky*(j-rf_radius))
                                dc=cos(kx*(i-rf_radius)+ ky*(j-rf_radius))

                                w1=w[ti,ni,wi+rf_area*ch]
                                    
                                cs+=w1*ds # response to sin/cos grating input
                                cc+=w1*dc

                                phi=atan2(cc,cs)  # phase to give max response

                                wi+=1

                        c=cs*cos(phi)+cc*sin(phi)     # max response

                        responses[ki,ai,ch,ni,ti]=c
    
    return responses

class pygroup(object):

    def save(self,g):
        import sys

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

   


class post_process_simulation(pygroup):
    def __init__(self,):
        self.save_attrs=[]
        self.save_data=[]
        self.name='post process simulation'
        self.sim=None
        self.neurons=None
        self.connections=None

    def apply(self):
        pass
        



class grating_response(post_process_simulation):
    
    def __init__(self,theta_mat=None,k_mat=None,print_time=True,verbose=False):

        self.verbose=verbose
        self.print_time=print_time
        self.name='grating response'
        
        if theta_mat is None:
            self.theta_mat=py.linspace(0.0,180,24)
        else:
            self.theta_mat=theta_mat
            
        if k_mat is None:
            self.k_mat=py.linspace(1,10,20)/13.0*py.pi            
        else:
            self.k_mat=k_mat
            
        self.save_attrs=['num_channels','rf_diameter']
        self.save_data=['k_mat','theta_mat','t','responses']



    def apply(self):
        self.monitor=self.sim.monitors['weights']

        t,weights=self.monitor.arrays()
        if len(weights.shape)==2:  # need to add another dimension
            weights.shape=(weights.shape[0],1,weights.shape[1])        
            
            
        num_neurons=weights.shape[1]
        c=self.monitor.container
        
        try:  # check for channel
            neurons=c.pre.neuron_list
        except AttributeError:
            neurons=[pre]

        self.num_channels=len(neurons)
        self.rf_diameter=neurons[0].rf_size
            
        if self.print_time:
            pn.utils.timeit(reset=True)
        self.responses=get_responses(t,weights,self.num_channels,self.rf_diameter,
                                     self.theta_mat,self.k_mat)
        self.t=t
        
        if self.print_time:
            print(pn.utils.timeit())
            
    @property
    def gratings(self):
        return get_gratings(self.rf_diameter,self.theta_mat,self.k_mat)



