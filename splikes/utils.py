from numpy import exp,concatenate,array,float,r_
from pylab import plot,ylabel,xlabel,gca,draw,legend,subplot,show,text,gcf,rand
from numpy import zeros,arange,ones,convolve,floor
import numpy as np

import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['figure.figsize'] = (10,8)
mpl.rcParams['axes.grid']=True

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
            
    finally:
        f.close()





def bigfonts(size=20,family='sans-serif'):
    
    from matplotlib import rc
    
    rc('font',size=size,family=family)
    rc('axes',labelsize=size)
    rc('axes',titlesize=size)
    rc('xtick',labelsize=size)
    rc('ytick',labelsize=size)
    rc('legend',fontsize=size)

bigfonts()

def running_average(t,y,T):
    N=len(t[t<=T])
    yf=np.convolve(y, np.ones((N,))/N,mode='same')
    return yf


def paramtext(x,y,*args,**kwargs):
        
    paramstr='\n'.join(args)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    T=text(x,y,paramstr,
       ha='center',
       va='top',
       bbox=props,
       transform=gca().transAxes,
       multialignment='left',
       **kwargs)
    

def plot_spike_lines(neuron,color,label):
    yl=gca().get_ylim()
    dyl=yl[1]-yl[0]
    count=0

    for t,n in neuron.saved_spikes:
        if count==0:
            plot([t,t],[yl[0],yl[0]+.1*dyl],color[n],linewidth=3,label=label)
        else:
            plot([t,t],[yl[0],yl[0]+.1*dyl],color[n],linewidth=3)
        count+=1

    print("Total number of %s spikes: %d " % (label,len(neuron.saved_spikes)))
    if neuron.N>1:
        for i in range(neuron.N):
            print("    Number of spikes for neuron %d: %d" % (i,len([t for t,n in neuron.saved_spikes if n==i])))
    
    
def timeplot(*args,**kwargs):
    import matplotlib.pyplot as plt 
    import matplotlib.ticker 
    
    def HMSFormatter(value, loc): 
        h = value // 3600 
        m = (value - h * 3600) // 60 
        s = value % 60 
        
        return "%02d:%02d:%02d" % (h,m,s) 

    def HMSFormatter2(value, loc): 
        h = value // 3600 
        m = (value - h * 3600) // 60 
        s = value % 60 
        ms=value%1
        
        return "%02d:%02d.%03d" % (m,s,ms*1000) 


    
    t=args[0]
    if max(t)<10:  # use ms
        gca().xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(HMSFormatter2)) 
    else:
        gca().xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(HMSFormatter)) 
    
    
    
    plot(*args,**kwargs) 
    gcf().autofmt_xdate()
    
ms=0.001
second=1000*ms
minute=60*second
hour=60*minute
day=24*hour
year=365.25*day
Hz=1.0

def plot_time_data(t,y,style='.-'):
    
    t=array(t)
    try:
        mx=t.max()
    except ValueError:  # empty t
        mx=0.0
        
    if mx>2*day:  # switch to days
        t=t/day
        unit='day'
    elif mx>2*hour:
        t=t/hour
        unit='hour'
    elif mx>2*minute:
        t=t/minute
        unit='min'
    elif mx>2*second:
        t=t/second
        unit='sec'
    else:
        unit='ms'

    try:
        plot(t,y,style)
    except ValueError:
        plot(t,[y],style)
        
    xlabel('Time (%s)' % unit)

def spike_counts(tmat,spikes,window=200):
    times,neurons=zip(*spikes)
    times=np.array(times)
    neurons=np.array(neurons)
    
    mx=np.max(neurons)
    
    N=mx+1
    
    counts=np.zeros( (N,len(tmat)-1) )
    
    for ti in range(len(tmat)-1):
        t1=tmat[ti]
        t2=tmat[ti+1]
    
        idx=np.where( (times<=t2) & (times>=t1) )[0]
        for ii in idx:
            counts[neurons[ii],ti]+=1
    
    return counts    
    

        



def average_rate_plot(monitors,window=200,neurons=[],xlim=None,ylim=None):

    if not isinstance(monitors,list):
        monitors=[monitors]
        
    if not isinstance(monitors[0],sp.SpikeMonitor):
        nn=monitors
        monitors=[]
        for n in nn:
            for m in n.monitors:
                if isinstance(m,SpikeMonitor):
                    monitors.append(m)

    for m in monitors:
        
        if xlim:
            t_min=xlim[0]
            t_max=xlim[1]
        else:
            t_min=m.t.min()
            t_max=m.t.max()
            
        if m.between:
            t=arange(m.between[0],m.between[1]+1)
            offset=m.between[0]
        else:
            t=arange(0,t_max+1)
            offset=0
            
        count=zeros(t.shape)
        if not neurons:
            for tt in m.t:
                count[int(tt)-offset]+=1
            N=m.cell.N
        else:
            for ss,tt in zip(m.spikes,m.t):
                if ss in neurons:
                    count[int(tt)-offset]+=1
            N=len(neurons)
        
        filt=ones(window)    
        rates=convolve(filt, count, mode='same')*1000.0/window/N
        
        plot_time_data(t,rates,'.-')
        ylabel('Rate (Hz)')
        
    
    
    

def raster_plot(monitors,xlim=None,ylim=None):
    
    if not isinstance(monitors,list):
        monitors=[monitors]
        
    if not isinstance(monitors[0],SpikeMonitor):
        neurons=monitors
        monitors=[]
        for n in neurons:
            for m in n.monitors:
                if isinstance(m,SpikeMonitor):
                    monitors.append(m)
        
    N=0
    for m in monitors:
        plot_time_data(m.t,m.spikes+N,'.')
    
        N+=m.cell.N
        
    ylabel('Neuron Number')

    ax=gca()
    if not ylim:
        ylim=ax.get_ylim()
        dy=ylim[1]-ylim[0]
        ylim=[ylim[0]-dy*.1, ylim[1]+dy*.1]
        
    ax.set_ylim(ylim)
    
    # we don't adjust the xlim like the ylim, so you can more easily compare with the states plots
    if xlim:
        ax.set_xlim(xlim)
        
        
    draw()
    
def plot_state(neuron,*args,**kwargs):
    
    items=kwargs.get('items',None)
    style=kwargs.get('style','.-')
    
    variable_names=args

    if not items is None:
        try:
            items[0]
        except TypeError:
            items=[items]

    data=[]
    for var in variable_names:
    
        for m in neuron.monitors:
            if m.var==var:
                t=m.t
                vals=m.vals
            
                if items is None:
                    plot_time_data(t,vals,style)
                    data.append( (t,vals) )
                else:
                    plot_time_data(t,vals[:,items],style)
                    data.append( (t,vals[:items]) )

    if len(variable_names)==1:
        ylabel(var)
        return data[0]
    else:
        ylabel('Value')
        legend(variable_names)
        return data
    

def plot_monitors(neuron):
    
    for i,m in enumerate(neuron.monitors):
        subplot(len(neuron.monitors),1,i+1)
        if isinstance(m,StateMonitor):
            plot_state(neuron,m.var)
        else:
            raster_plot(neuron,[0,max(m.t)])
    show()
    draw()
    

def sq(x,width):
    y=zeros(x.shape)
    y[abs(x)<=(width/2.0)]=1.0
    return y

def make_square(N=10,sz=100,rates=[5,55],width=10,display=False):
    min_rate,max_rate=rates

    try:
        min_width,max_width=width
    except TypeError:
        min_width,max_width=width,width
    
    centers=(r_[0:N]+0.5)*sz/N

    # reverse, so the shift works easier down below
    centers=sz-centers
    
    mid=sz/2
    idx=r_[0:sz]-mid
    
    
    l=[]
    for c in centers:

        width=rand()*(max_width-min_width)+min_width

        g=sq(idx,width)*(max_rate-min_rate)+min_rate
        g=concatenate((g[mid:],g[0:mid]))
        
        r=concatenate((g[c:],g[0:c]))
        l.append(r)
        
    a=array(l,float)
        
    if display:
        
        for r in a:
            plot(r,'.-')
    
    return a


def make_gaussian(N=10,sz=100,rates=[5,55],sigma=10,display=False):
    
    min_rate,max_rate=rates

    try:
        min_sigma,max_sigma=sigma
    except TypeError:
        min_sigma,max_sigma=sigma,sigma
    
    centers=(r_[0:N]+0.5)*sz/N

    # reverse, so the shift works easier down below
    centers=sz-centers
    
    mid=sz/2
    idx=r_[0:sz]-mid
    
    
    l=[]
    for c in centers:

        sigma=rand()*(max_sigma-min_sigma)+min_sigma

        g=exp(-idx**2/(2.0*sigma**2))*(max_rate-min_rate)+min_rate
        g=concatenate((g[mid:],g[0:mid]))
        
        r=concatenate((g[c:],g[0:c]))
        l.append(r)
        
    a=array(l,float)
        
    if display:
        
        for r in a:
            plot(r,'.-')
    
    return a


def convert_neuron_equations(D):
    import re
    loop_var='__i'

    equation_lines=[]

    for line in D['equations']:
        line=line.replace(' ','')

        line=line.split('#')[0]  # get rid of comment
        if not line:
            continue

        if ':' in line:
            line,shape_str=line.split(':')  # get shape info
        else:
            shape_str=''

        parts=line.split('=')
        left_side=parts[0]

        if '/' in left_side:  # derivative
            vpart=left_side.split('/')
            varstr=vpart[0][1:].strip()
            op_str='+=sim.dt*'
        else:  # equality
            varstr=left_side
            op_str='='

        varstr="%s[%s]%s" % (varstr, loop_var,op_str)

        line=parts[1]

        # put the indexing code in
        for var in D['variables']:            
            pstr=r'\b%s\b' % var  # match words (\b = word boundary)
            pattern=re.compile(pstr)

            line,count=pattern.subn('%s[%s]' % (var,loop_var),line)

        # put the parameters in
        for var in D['parameters']:            
            pstr=r'\b%s\b' % var  # match words (\b = word boundary)
            pattern=re.compile(pstr)

            # TODO: this should be done with a buffer
            if not isinstance(D['parameters'][var],(int,float)):
                line,count=pattern.subn('%s[%s]' % (var,loop_var),line)
            else:
                line,count=pattern.subn('%s' % var,line)

        line=varstr+"("+line+")"

        equation_lines.append(line)

    flag=False
    indent='    '
    s="for %s in range(self.N):\n" % loop_var
    for line in equation_lines:
        s+=indent+line+"\n"

    return s


def translate_neuron(neuron_str):

    def indent(s,num):

        if num>0:
            indent_str=' '*num
            lines=s.split('\n')
            new_lines=[]
            for line in lines:
                new_lines.append(indent_str+line)

            s2='\n'.join(new_lines)
            return s2
        elif num<0:
            num=-num    
            lines=s.split('\n')
            new_lines=[]
            for line in lines:
                new_lines.append(line[num:])

            s2='\n'.join(new_lines)
            return s2
        else:
            return s


    import yaml
    
    D=yaml.load(neuron_str)
    D['parameter lines']=D['parameters']
    del D['parameters']
    
    code_str=""
    
    code_str+="""
from splikes cimport *
cimport cython
import pylab

import numpy as np
cimport numpy as np
    """    
    code_str+="\n"
        
    variables={}
    for p in D['equations']:
        parts=p.strip().split('=')
        left=parts[0].strip()
        if '/' in left:
            varname=left.split('/')[0].strip()[1:]
        else:
            varname=left
            
            
        variables[varname]=p
    
    D['variables']=variables 
    
    parameters={}
    for p in D['parameter lines']:
        parts=p.strip().split('=')
        varname=parts[0].strip()
        value=parts[1].strip()
        parameters[varname]=eval(value)
        
    D['parameters']=parameters
    
    code_str+="cdef class {name}(neuron):\n".format(**D)
    
    code_str+=indent("cdef public double "+",".join(parameters.keys()),4)+"\n"
    code_str+="    cdef public np.ndarray "+",".join(variables.keys())+"\n"
    
    code_str+="    cpdef _reset(self):\n"
    
    for varname in variables:
        if varname in ['rate']:
            continue    

        code_str+="        self.%s=np.zeros(self.N,dtype=np.float)\n" % varname
    
    code_str+="        neuron._reset(self)\n"
    
    code_str+="""
    def __init__(self,N):
        neuron.__init__(self,N)
    """
    code_str+="\n"
    for p in parameters:
        code_str+="        self.%s=%s\n" % (p,str(parameters[p]))
    
    code_str+="        self._reset()\n"
    
    
    code_str+="""
    @cython.cdivision(True)
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    cpdef update(self,double t,simulation sim):
        cdef int __i,__j
        cdef connection c
        cdef neuron pre
    """
    code_str+="\n"
    for varname in variables:
        code_str+="        cdef double *%s=<double *>self.%s.data\n" % (varname,varname)
    for varname in parameters:
        code_str+="        cdef double %s=self.%s\n" % (varname,varname)
    
    
    code_str+="""
        cdef double *W,*state
        cdef double spike_scale
        cdef int *spiking   
    """
    
    
    code_str+="""
        for c in self.connections_pre:
            pre=c.pre
            W=c.W
            spiking=<int *>pre.spiking.data
            spike_scale=c.spike_scale
            
            if pre.is_spike and c.use_state:
                state=<double *>c.state.data
                for __j in range(pre.N):
                    if spiking[__j]:
                        for __i in range(self.N):
                            state[__i]+=spike_scale*W[__i*pre.N+__j]    
    """
    
    code_str+="\n"
    code_str+=indent(convert_neuron_equations(D),8)
    
    spiking=D['spiking'].strip()
    if 'threshold' in spiking:
        var_threshold=spiking.split('>')[0].strip()
        if var_threshold not in variables:
            raise ValueError("%s not in variables %s" % (var_threshold,str(variables)))
    
        if 'reset' not in parameters:
            raise ValueError("'reset' not in parameters")
        if 'threshold' not in parameters:
            raise ValueError("'threshold' not in parameters")
    
        code_str+="""
        spiking=<int *>self.spiking.data
        self.is_spike=0
        for __i in range(self.N):
            if %s[__i]>self.threshold:
                spiking[__i]=1
                self.is_spike=1
                self.post_count+=1
                if self.save_spikes_begin<=t<=self.save_spikes_end:
                    self.saved_spikes.append( (t,__i) )
                %s[__i]=self.reset
            else:
                spiking[__i]=0            
    """ % (var_threshold,var_threshold)
    elif 'poisson' in spiking:
        if 'rate' not in variables:
            raise ValueError("'rate' not in variables")
        
        code_str+="""
        spiking=<int *>self.spiking.data
        self.is_spike=0
        for __i in range(self.N):
                
            if randu()<(rate[__i]*sim.dt):
                spiking[__i]=1
                self.is_spike=1
                self.post_count+=1
                if self.save_spikes_begin<=t<=self.save_spikes_end:
                    self.saved_spikes.append( (t,__i) )
            else:
                spiking[__i]=0
        """
    else:
        if spiking:
            raise ValueError("Spiking '%s' not implemented" % spiking)

    
    
    
    return code_str


def convert_connection_equations(D):
    import re
    loop_var1='__i'
    loop_var2='__j'

    equation_lines=[]
    shape_lines=[]
    if not D['equations']:
        return ''

    for eqn in D['equations']:
        line=eqn.keys()[0]
        shape=eqn[line]

        line=line.replace(' ','')

        line=line.split('#')[0]  # get rid of comment
        if not line:
            continue

        parts=line.split('=')
        left_side=parts[0]

        if '/' in left_side:  # derivative
            vpart=left_side.split('/')
            varstr=vpart[0][1:].strip()
            op_str='+=sim.dt*'
        else:  # equality
            varstr=left_side
            op_str='='

        if shape=='pre':
            varstr="%s[%s]%s" % (varstr, loop_var2,op_str)
        elif shape=='post':
            varstr="%s[%s]%s" % (varstr, loop_var1,op_str)
        else:
            varstr="%s[__wi]%s" % (varstr,op_str)


        shape_lines.append(shape)
        line=parts[1]

        for var in D['variables']:
            shape=D['variables'][var][1]
            pstr=r'\b%s\b' % var  # match words (\b = word boundary)
            pattern=re.compile(pstr)

            if shape=='pre':
                line,count=pattern.subn('%s[%s]' % (var,loop_var2),line)
            elif shape=='post':
                line,count=pattern.subn('%s[%s]' % (var,loop_var1,),line)
            else:
                line,count=pattern.subn('%s[__wi]' % (var,),line)

        var='pre'
        pstr=r'\b%s\b' % var  # match words (\b = word boundary)
        pattern=re.compile(pstr)

        line,count=pattern.subn('%s[%s]/sim.dt' % (var,loop_var2),line)  # divide by dt

        var='post'
        pstr=r'\b%s\b' % var  # match words (\b = word boundary)
        pattern=re.compile(pstr)

        line,count=pattern.subn('%s[%s]/sim.dt' % (var,loop_var1),line)


        # put the parameters in
        for var in D['parameters']:            
            pstr=r'\b%s\b' % var  # match words (\b = word boundary)
            pattern=re.compile(pstr)

            if not isinstance(D['parameters'][var],(int,float)):
                line,count=pattern.subn('%s[%s]' % (var,loop_var2),line)
            else:
                line,count=pattern.subn('%s' % var,line)

        line=varstr+"("+line+")"

        equation_lines.append(line)

    old_shape='nothing'

    s=''
    for line,shape in zip(equation_lines,shape_lines):
        if shape!=old_shape:
            if shape=='pre':
                s+="for %s in range(self.pre.N):\n" % loop_var2
                indent='    '
            elif shape=='post':
                s+="for %s in range(self.post.N):\n" % loop_var1
                indent='    '
            else:
                s+="for %s in range(self.post.N):\n    for %s in range(self.pre.N):\n" % (loop_var1,loop_var2)
                s+="        __wi=%s*self.pre.N+%s\n" % (loop_var1,loop_var2)
                indent='    '*2

            old_shape=shape

        s+=indent+line+"\n"

    return s



def translate_connection(connection_str):
    def indent(s,num):

        if num>0:
            indent_str=' '*num
            lines=s.split('\n')
            new_lines=[]
            for line in lines:
                new_lines.append(indent_str+line)

            s2='\n'.join(new_lines)
            return s2
        elif num<0:
            num=-num    
            lines=s.split('\n')
            new_lines=[]
            for line in lines:
                new_lines.append(line[num:])

            s2='\n'.join(new_lines)
            return s2
        else:
            return s


    import yaml


    D=yaml.load(connection_str)
    D['parameter lines']=D['parameters']
    del D['parameters']
    
    code_str=""
    variables={}
    new_equations=[]
    for p in D['equations']:
        try:
            eqn=p.keys()[0]
            shape=p[eqn]
            shape=shape.split('#')[0]  # get rid of comments
            new_equations.append({eqn:shape})
        except AttributeError:
            eqn=p
            eqn=eqn.split('#')[0] # get rid of comments
            shape='full'
            new_equations.append({eqn:shape})
        
        
        parts=eqn.strip().split('=')
        left=parts[0].strip()
        if '/' in left:
            varname=left.split('/')[0].strip()[1:]
        else:
            varname=left
            
        if varname in ['W','post_rate','pre_rate']:
            continue
            
        variables[varname]=eqn,shape
    
    D['equations']=new_equations    
    D['variables']=variables 
    
    parameters={}
    for p in D['parameter lines']:
        parts=p.strip().split('=')
        varname=parts[0].strip()
        value=parts[1].strip()
        parameters[varname]=eval(value)
        
    D['parameters']=parameters
    
    
    code_str=""
    
    code_str+="""
from splikes cimport *
cimport cython
import pylab

import numpy as np
cimport numpy as np
    """
    code_str+="\n"
    
    # put in the original translated string
    lines2=connection_str.split('\n')
    lines2='\n'.join(['# ' + line for line in lines2])
    code_str+="\n"+lines2+"\n"
    
    
    code_str+="cdef class {name}(connection):\n".format(**D)
    
    code_str+=indent("cdef public double "+",".join(parameters.keys()),4)+"\n"
    code_str+="    cdef public np.ndarray "+",".join(variables.keys())+"\n"
    
    code_str+="    cpdef _reset(self):\n"
    
    for varname in variables:
        shape=variables[varname][1]
        if shape=='pre':
            code_str+="        self.%s=np.zeros(self.pre.N,dtype=np.float)\n" % varname
        elif shape=='post':
            code_str+="        self.%s=np.zeros(self.post.N,dtype=np.float)\n" % varname
        elif shape=='full':
            code_str+="        self.%s=np.zeros( (self.post.N,self.pre.N),dtype=np.float)\n" % varname
        else:
            raise ValueError('Illegal shape: %s' % shape)
    
    code_str+="        connection._reset(self)\n"
    
    
    code_str+="""
    def __init__(self,neuron pre,neuron post,initial_weight_range=None,state=None):
        connection.__init__(self,pre,post,initial_weight_range,state)
    """
    code_str+="\n"
    
    for p in parameters:
        code_str+="        self.%s=%s\n" % (p,str(parameters[p]))
    
    code_str+="        self._reset()\n"
    
    code_str+="""
    @cython.cdivision(True)
    @cython.boundscheck(False) # turn of bounds-checking for entire function
    cpdef update(self,double t,simulation sim):
        cdef int __i,__j
    """
    code_str+="\n"
    for varname in variables:
        code_str+="        cdef double *%s=<double *>self.%s.data\n" % (varname,varname)
    for varname in parameters:
        code_str+="        cdef double %s=self.%s\n" % (varname,varname)
    
        
    
    code_str+="""
        cdef double *W=self.W
        cdef double *post_rate=<double *>self.post.rate.data
        cdef double *pre_rate=<double *>self.pre.rate.data
        cdef int *pre,*post   # spikes for pre and post
        cdef int __wi
        
        
        pre=<int *>self.pre.spiking.data
        post=<int *>self.post.spiking.data
    """
        
    code_str+="\n"
    code_str+=indent(convert_connection_equations(D),8)
        
    code_str+="\n"
    code_str+=indent("self.apply_weight_limits()\n",8)
        
    return code_str
    
    
    