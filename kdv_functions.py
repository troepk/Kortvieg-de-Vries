import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.ticker import MaxNLocator
import scipy.integrate as integrate
import scipy.optimize as optimize
import scipy.misc as misc
import math

from numba import jit
from tqdm import tqdm
from matplotlib import animation
from IPython.display import HTML

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")




# ---------------------------- Functions ----------------------------
# Defining the initial function
@jit
def func(x,N):
    
    result = (N*(N+1))/(np.cosh(x)**2)
    
    return result


# Stepifying the initial function.
def u_0(x,N,spacesteps):
    
    xrange= list(np.arange(-(spacesteps+1)*x,(spacesteps+1)*x,x))
    result = list(map(lambda x: func(x,N),xrange))
    
    return result


# Calculating timestep from spacestep 
def Time_step(x,N):
    
    umax = func(0,N)
    result = 0.9*2/(3*np.sqrt(3)*abs(-2*umax+1/(x**2))) *x
    
    return result


# Defining a parametrised function, which solves kdv
def func_para(x,t,N,B):
    
    result = (N*(N+1))/(np.cosh(x-B*t)**2)
    
    return result



# Depth profile
def DEPTH(x,h_0,L ,x_shift):
    
    x = x-x_shift
    
    if x > 0 and x < L:
        
        result = 0.5*(1+h_0+(1-h_0)*np.cos(np.pi*x/L))
        
    elif x >= L:
        
        result = h_0
    
    else:
        result=1
        
    return result
'''
def DEPTH(x,h_0,L ,x_shift):
    
    
    if x > x_shift and x < x_shift+L:
        
        result = 0.5*(1+h_0+(1-h_0)*np.cos(np.pi*(x-x_shift)/L))
        
    elif x < -x_shift and x > -(x_shift+L):
        
        result = 0.5*(1+h_0+(1-h_0)*np.cos(np.pi*(x+x_shift)/L))
        
    elif x >= x_shift+L or x <= -(x_shift+L):
        
        result = h_0
    
    else:
        result=1
        
    return result
'''

# Depth profile
def DEPTH_hill(x,h_0,L ,x_shift):
    
    x = x-x_shift
    
    result = 1-h_0*np.exp(-(x/L)**2)
        
    return result
# ---------------------------- Functions ----------------------------




# ---------------------------- Algorithm h=const. ----------------------------
# Applying the FirstStepCalc function to all positions
def FirstStep(start,t,x):
    
    result =[]
    
    l = len(start)
    
    for i in range(l):
      
        A = start[i]
        B = t/x *(start[(i-1)%l]+start[i]+start[(i+1)%l])*(start[(i+1)%l]-start[(i-1)%l])
        C = t/(2*x**3) *(start[(i+2)%l]-2*(start[(i+1)%l]-start[(i-1)%l])-start[(i-2)%l])
    
        first_step_i = A-B-C
    
        result.append(first_step_i)
        
    return result


# Applying the NextStepCalc function to all positions
def NextStep(c,p,t,x):
    
    result =[]
    
    l = len(c)
    
    for i in range(l):
      
        A = p[i]
        B = 2*(t/x) *(c[(i-1)%l]+c[i%l]+c[(i+1)%l])*(c[(i+1)%l]-c[(i-1)%l])
        C = t/x**3 *(c[(i+2)%l]-2*(c[(i+1)%l]-c[(i-1)%l])-c[(i-2)%l])
    
        next_step_i = A-B-C   
        
        result.append(next_step_i)
        
    return result


def FirstStep_DEPTH_hill(start,t,x,h_0,L,x_shift):
    
    result =[]
    
    l = len(start)
    
    for i in range(l):
      
        h =  DEPTH_hill(x*i,h_0,L,x_shift)
    
        A = start[i]
        B = t/(x*h**(7/4)) *(start[(i-1)%l]+start[i]+start[(i+1)%l])*(start[(i+1)%l]-start[(i-1)%l])
        C = t/(2*x**3) *(start[(i+2)%l]-2*(start[(i+1)%l]-start[(i-1)%l])-start[(i-2)%l])*h**(1/2)
    
        first_step_i = A-B-C
    
        result.append(first_step_i)
        
    return result


# Calculating one u in the next timestep at one position i
# current, previous, timestep, spacestep, positionindex
@jit
def NextStep_DEPTH_hill(c,p,t,x,h_0,L,x_shift):
    
    result =[]
    
    l = len(c)
    
    for i in range(l):
        
        h =  DEPTH_hill(x*i,h_0,L,x_shift)
    
        A = p[i]
        B = 2*(t/x*h**(7/4)) *(c[(i-1)%l]+c[i]+c[(i+1)%l])*(c[(i+1)%l]-c[(i-1)%l])
        C = t/x**3 *(c[(i+2)%l]-2*(c[(i+1)%l]-c[(i-1)%l])-c[(i-2)%l])*h**(1/2)
    
        next_step_i = A-B-C   
        
        result.append(next_step_i)
    
    return result


def FirstStep_DEPTH(start,t,x,h_0,L,x_shift):
    
    result =[]
    
    l = len(start)
    
    for i in range(l):
      
        h =  DEPTH(x*i,h_0,L,x_shift)
    
        A = start[i]
        B = t/(x*h**(7/4)) *(start[(i-1)%l]+start[i]+start[(i+1)%l])*(start[(i+1)%l]-start[(i-1)%l])
        C = t/(2*x**3) *(start[(i+2)%l]-2*(start[(i+1)%l]-start[(i-1)%l])-start[(i-2)%l])*h**(1/2)
    
        first_step_i = A-B-C
    
        result.append(first_step_i)
        
    return result


# Calculating one u in the next timestep at one position i
# current, previous, timestep, spacestep, positionindex
@jit
def NextStep_DEPTH(c,p,t,x,h_0,L,x_shift):
    
    result =[]
    
    l = len(c)
    
    for i in range(l):
        
        h =  DEPTH(x*i,h_0,L,x_shift)
    
        A = p[i]
        B = 2*(t/x*h**(7/4)) *(c[(i-1)%l]+c[i]+c[(i+1)%l])*(c[(i+1)%l]-c[(i-1)%l])
        C = t/x**3 *(c[(i+2)%l]-2*(c[(i+1)%l]-c[(i-1)%l])-c[(i-2)%l])*h**(1/2)
    
        next_step_i = A-B-C   
        
        result.append(next_step_i)
    
    return result
# ---------------------------- Algorithm h=const. ----------------------------




# ---------------------------- Apply ZabuskyKruskal, h=const. ----------------------------
def ZabuskyKruskal( N, time,spacesteps, t, x,**kwargs):
    
    step_in_time = kwargs.get('step_in_time',100)
    
    initial_array = u_0(x,N,spacesteps)
    allData = np.array([initial_array])
    
    step = FirstStep(initial_array,t,x)
    allData = np.append(allData,[step],axis=0)
    
    allTimes = np.array([0,t])
    
    current_step, prev_step = step, initial_array
    
    for count in tqdm(range(2,time)):

        step = NextStep(current_step, prev_step,t,x)
        
        if count%step_in_time==0:
            allData = np.append(allData,[step],axis=0)
            allTimes = np.append(allTimes,[count*t],axis=0)
            
        current_step, prev_step = step, current_step
                
    return allData, allTimes
# ---------------------------- Apply ZabuskyKruskal, h=const. ----------------------------




# ---------------------------- Apply ZabuskyKruskal, h profile ----------------------------
def ZabuskyKruskal_DEPTH( N, time,spacesteps, t, x,h_0,L,**kwargs):
    
    step_in_time = kwargs.get('step_in_time',100)
    x_shift = kwargs.get('x_shift',0)
    
    initial_array = u_0(x,N,spacesteps)
    allData = np.array([initial_array])
    
    step = FirstStep_DEPTH(initial_array,t,x,h_0,L,x_shift)
    allData = np.append(allData,[step],axis=0)
    
    allTimes = np.array([0,t])
    
    current_step, prev_step = step, initial_array
    
    for count in tqdm(range(2,time)):

        step = NextStep_DEPTH(current_step, prev_step,t,x,h_0,L,x_shift)
        
        if count%step_in_time==0:
            allData = np.append(allData,[step],axis=0)
            allTimes = np.append(allTimes,[count*t],axis=0)
            
        current_step, prev_step = step, current_step
                
    return allData, allTimes


def ZabuskyKruskal_DEPTH_hill( N, time,spacesteps, t, x,h_0,L,**kwargs):
    
    step_in_time = kwargs.get('step_in_time',100)
    x_shift = kwargs.get('x_shift',0)
    
    initial_array = u_0(x,N,spacesteps)
    allData = np.array([initial_array])
    
    step = FirstStep_DEPTH_hill(initial_array,t,x,h_0,L,x_shift)
    allData = np.append(allData,[step],axis=0)
    
    allTimes = np.array([0,t])
    
    current_step, prev_step = step, initial_array
    
    for count in tqdm(range(2,time)):

        step = NextStep_DEPTH_hill(current_step, prev_step,t,x,h_0,L,x_shift)
        
        if count%step_in_time==0:
            allData = np.append(allData,[step],axis=0)
            allTimes = np.append(allTimes,[count*t],axis=0)
            
        current_step, prev_step = step, current_step
                
    return allData, allTimes
# ---------------------------- Apply ZabuskyKruskal, h profile ----------------------------




# ---------------------------- Animation ----------------------------
def Animator(allData,**kwargs):
    
    step_in_time = kwargs.get('step_in_time',100)
    ani_interval = kwargs.get('ani_interval',100)  
    
    fig = plt.figure()
    ims = []

    for i in tqdm(range(0,len(allData),1)):
    
        if i%step_in_time==0:
            im = plt.plot(allData[i],color='b')
            ims.append(im)
    
    ani = animation.ArtistAnimation(fig, ims, interval=ani_interval, blit=True)
    return HTML(ani.to_html5_video())
# ---------------------------- Animation ----------------------------




# ---------------------------- Determining the Integrals ----------------------------
def INTEGRATER(N, allData, allTimes, xrange,**kwargs):
    
    step_in_time = kwargs.get('step_in_time',100)
    
    integrals = np.array([])
    umaxs = np.array([])
    
    for i in tqdm(range(0, len(allData))):
    
        current_time = allTimes[i]
        
        def f(x,B):
        
            result = func_para(x,current_time,N,B)
            return result
        
        popt, pcov = optimize.curve_fit(f,xrange,allData[i], maxfev=10000)

        P_0 = integrate.quad(lambda x: f(x,popt[0]),-math.inf, math.inf)[0]

        P_1 = integrate.quad(lambda x: f(x,popt[0])**2,-math.inf, math.inf)[0]

        def f_new(x):
            return f(x,popt[0])

        def f_derivative(x):
            return misc.derivative(f_new,x,dx=1e-8)

        P_2 = integrate.quad(lambda x: 2*f(x,popt[0])**3-(f_derivative(x))**2,-math.inf, math.inf)[0]

        u_max = np.amax(allData[i])
        
        if i==0:
            
            integrals =np.array([[P_0,P_1,P_2,current_time,u_max]])
            umaxs = np.array([[u_max,current_time]])
            
        else:
            
            integrals = np.append(integrals,[[P_0,P_1,P_2,current_time,u_max]],axis=0)
            umaxs = np.append(umaxs,[[u_max,current_time]],axis=0)
    
    return integrals.T, umaxs.T


def INTEGRATER_2(allData, allTimes, x,deltaT):
    
    integrals = np.array([])
    umaxs = np.array([])
    
    for i in tqdm(range(0,len(allData))):
        
        current_time = allTimes[i]
        
        space_dim = len(allData[i])
        
        p_0 = np.array([0])
        p_1 = np.array([0])
        p_2 = np.array([0])
        
        for j in range(0,space_dim):
            
            y_current = allData[i,j]
            y_next = allData[i,(j+1)%space_dim]
            
            y = 0.5 * (y_current + y_next)
            
            dy_dx = (y_next - y_current)/x
            
            p_0_bar = y_current * x
            p_1_bar = y_current**2 * x
            p_2_bar = (2*y_current**3 - dy_dx**2) * x
            
            p_0 = np.append(p_0,[p_0_bar],axis=None) 
            p_1 = np.append(p_1,[p_1_bar],axis=None) 
            p_2 = np.append(p_2,[p_2_bar],axis=None) 
        
        P_0 = np.sum(p_0)
        P_1 = np.sum(p_1)
        P_2 = np.sum(p_2)
        
        u_max = np.amax(allData[i])
        
        if i==0:
            
            integrals = np.array([[P_0,P_1,P_2,current_time,u_max]])
            umaxs = np.array([[u_max,current_time]])
            
        else:
            
            integrals = np.append(integrals,[[P_0,P_1,P_2,current_time,u_max]],axis=0)
            umaxs = np.append(umaxs,[[u_max,current_time]],axis=0)
        
    return integrals.T, umaxs.T
# ---------------------------- Determining the Integrals ----------------------------




# ----------------------------  Plotting ----------------------------
def Plotter_Integrals(allData_1,allTimes_1,integrals_1,xrange,deltaT,name,**kwargs):
    
    timerange = kwargs.get('timerange',[allTimes_1[0],allTimes_1[-1]])
    timeres = kwargs.get('timeres',8000) 
    
    plot_time_steps = [0]
    plot_data_steps = [allData_1[0]]

    for item,data in zip(allTimes_1,allData_1):

        if item-plot_time_steps[-1] >=deltaT*timeres and item>=timerange[0] and item<=timerange[1]:

            plot_time_steps.append(item)
            plot_data_steps.append(data)
        
    plt.rcParams.update({'font.size': 20})
    
    fig, axs = plt.subplots(3,figsize=(16,9),sharex='col')
    fig.tight_layout(pad=3)

    colors = plt.cm.plasma(np.linspace(0,1,len(plot_time_steps)))
    
    axs[0].plot(integrals_1[3],integrals_1[0],color ='black')
    axs[1].plot(integrals_1[3],integrals_1[1],color ='black')
    axs[2].plot(integrals_1[3],integrals_1[2],color ='black')
    
    # Title
    axs[0].set_title('Integral $P_0$', size=20)
    axs[1].set_title('Integral $P_1$', size=20)
    axs[2].set_title('Integral $P_2$', size=20)
    
    # Tick params
    axs[0].minorticks_on()
    axs[1].minorticks_on()
    axs[2].minorticks_on()
    
    axs[0].tick_params(labelsize=20)
    axs[1].tick_params(labelsize=20)
    axs[2].tick_params(labelsize=20)
    
    axs[0].tick_params(direction='in',which='minor',length=5,bottom=True,top=True,left=True,right=True)
    axs[1].tick_params(direction='in',which='minor',length=5,bottom=True,top=True,left=True,right=True)
    axs[2].tick_params(direction='in',which='minor',length=5,bottom=True,top=True,left=True,right=True)
    
    # grid
    axs[0].grid()
    axs[1].grid()
    axs[2].grid()
    
    # xlabel
    axs[2].set_xlabel('Dimensionless time',fontsize=20)
    
    
    count = 0
    for item,data,color in zip(plot_time_steps,plot_data_steps,colors):

        axs[0].axvline(x=item, color =color)
        axs[1].axvline(x=item, color =color)
        axs[2].axvline(x=item, color =color)

    plt.savefig(str(name)+'.pdf',format='pdf')
    plt.show()
    
    
def Plotter_Waves(allData_1,allTimes_1,integrals_1,xrange,deltaT,name,**kwargs):
    
    timerange = kwargs.get('timerange',[allTimes_1[0],allTimes_1[-1]])
    positionrange = kwargs.get('positionrange',[xrange[0],xrange[-1]])
    timeres = kwargs.get('timeres',8000) 
    
    plot_time_steps = [0]
    plot_data_steps = [allData_1[0]]

    for item,data in zip(allTimes_1,allData_1):

        if item-plot_time_steps[-1] >=deltaT*timeres and item>=timerange[0] and item<=timerange[1]:

            plot_time_steps.append(item)
            plot_data_steps.append(data)

    plt.clf()
    plt.rcParams.update({'font.size': 20})
    
    fig, axs = plt.subplots(len(plot_time_steps),figsize=(16,9),sharex='col')
    fig.tight_layout()

    colors = plt.cm.plasma(np.linspace(0,1,len(plot_time_steps)))
    
    count = 0
    for item,data,color in zip(plot_time_steps,plot_data_steps,colors):

        axs[count].set_xlim(positionrange[0],positionrange[1])
        axs[count].set_yscale('symlog')
        axs[count].plot(xrange,data, color =color)
        axs[count].set_title('Time: '+str(round(item,2)), size=20)
        axs[count].minorticks_on()
        axs[count].tick_params(labelsize=20)
        axs[count].tick_params(direction='in',which='minor',length=5,bottom=True,top=True,left=True,right=True)
        axs[count].grid()
        axs[count].set_ylabel('u')
        
        count = count +1
    
    axs[-1].set_xlabel('Dimensionless positon $3(x-vt)/d$',fontsize=20)
    plt.savefig(str(name)+'.pdf',format='pdf')
    plt.show()
    
    
def Plotter_Waves_DEPTH(allData_1,allTimes_1,integrals_1,xrange,deltaT,name,h_0,L,**kwargs):
    
    timerange = kwargs.get('timerange',[allTimes_1[0],allTimes_1[-1]])
    positionrange = kwargs.get('positionrange',[xrange[0],xrange[-1]])
    timeres = kwargs.get('timeres',8000) 
    x_shift = kwargs.get('x_shift',0)
    
    plot_time_steps = [0]
    plot_data_steps = [allData_1[0]]

    for item,data in zip(allTimes_1,allData_1):

        if item-plot_time_steps[-1] >=deltaT*timeres and item>=timerange[0] and item<=timerange[1]:

            plot_time_steps.append(item)
            plot_data_steps.append(data)

    plt.clf()
    plt.rcParams.update({'font.size': 20})
    fig, axs = plt.subplots(len(plot_time_steps)+1,figsize=(16,9),sharex='col')
    fig.tight_layout()

    colors = plt.cm.plasma(np.linspace(0,1,len(plot_time_steps)))
    
    h_data = list(map(lambda x: -DEPTH(x,h_0,L,x_shift),xrange))
    count = 0
    for i in range(len(plot_data_steps)):

        item = plot_time_steps[i]
        data = plot_data_steps[i]
        color = colors[i]
        
        axs[count].set_xlim(positionrange[0],positionrange[1])
        axs[count].set_yscale('symlog')
        #axs[count].plot(xrange,h_data, color ='black')
        axs[count].plot(xrange,data, color =color)
        axs[count].set_title('Time: '+str(round(item,2)), size=20)
        axs[count].minorticks_on()
        axs[count].tick_params(labelsize=20)
        axs[count].tick_params(direction='in',which='minor',length=5,bottom=True,top=True,left=True,right=True)
        axs[count].grid()
        axs[count].set_ylabel('u')
        
        count = count +1
    
    axs[-1].set_xlim(positionrange[0],positionrange[1])
    axs[-1].plot(xrange,h_data, color ='black')
    axs[-1].set_title('Depth profile', size=20)
    axs[-1].minorticks_on()
    axs[-1].tick_params(labelsize=20)
    axs[-1].tick_params(direction='in',which='minor',length=5,bottom=True,top=True,left=True,right=True)
    axs[-1].grid()
    axs[-1].set_ylabel('h')
    
    axs[-1].set_xlabel('Dimensionless positon $3(x-vt)/d$',fontsize=20)
    plt.savefig(str(name)+'.pdf',format='pdf')
    plt.show()
    
    
def Plotter_Waves_DEPTH_hill(allData_1,allTimes_1,integrals_1,xrange,deltaT,name,h_0,L,**kwargs):
    
    timerange = kwargs.get('timerange',[allTimes_1[0],allTimes_1[-1]])
    positionrange = kwargs.get('positionrange',[xrange[0],xrange[-1]])
    timeres = kwargs.get('timeres',8000) 
    x_shift = kwargs.get('x_shift',0)
    
    plot_time_steps = [0]
    plot_data_steps = [allData_1[0]]

    for item,data in zip(allTimes_1,allData_1):

        if item-plot_time_steps[-1] >=deltaT*timeres and item>=timerange[0] and item<=timerange[1]:

            plot_time_steps.append(item)
            plot_data_steps.append(data)

    plt.clf()
    plt.rcParams.update({'font.size': 20})
    fig, axs = plt.subplots(len(plot_time_steps)+1,figsize=(16,9),sharex='col')
    fig.tight_layout()

    colors = plt.cm.plasma(np.linspace(0,1,len(plot_time_steps)))
    
    h_data = list(map(lambda x: -DEPTH_hill(x,h_0,L,x_shift),xrange))
    count = 0
    for i in range(len(plot_data_steps)):

        item = plot_time_steps[i]
        data = plot_data_steps[i]
        color = colors[i]
        
        axs[count].set_xlim(positionrange[0],positionrange[1])
        axs[count].set_yscale('symlog')
        #axs[count].plot(xrange,h_data, color ='black')
        axs[count].plot(xrange,data, color =color)
        axs[count].set_title('Time: '+str(round(item,2)), size=20)
        axs[count].minorticks_on()
        axs[count].tick_params(labelsize=20)
        axs[count].tick_params(direction='in',which='minor',length=5,bottom=True,top=True,left=True,right=True)
        axs[count].grid()
        axs[count].set_ylabel('u')
        
        count = count +1
    
    axs[-1].set_xlim(positionrange[0],positionrange[1])
    axs[-1].plot(xrange,h_data, color ='black')
    axs[-1].set_title('Depth profile', size=20)
    axs[-1].minorticks_on()
    axs[-1].tick_params(labelsize=20)
    axs[-1].tick_params(direction='in',which='minor',length=5,bottom=True,top=True,left=True,right=True)
    axs[-1].grid()
    axs[-1].set_ylabel('h')
    
    axs[-1].set_xlabel('Dimensionless positon $3(x-vt)/d$',fontsize=20)
    plt.tight_layout()
    plt.savefig(str(name)+'.pdf',format='pdf')
    plt.show()
    

def Plot_u_time(umax,allTimes_1,name,deltaT,**kwargs):
   
    timerange = kwargs.get('timerange',[allTimes_1[0],allTimes_1[-1]])
    timeres = kwargs.get('timeres',8000) 
    
    plot_time_steps = [0]
    
    for item in allTimes_1:

        if item-plot_time_steps[-1] >=deltaT*timeres and item>=timerange[0] and item<=timerange[1]:

            plot_time_steps.append(item)
            
    colors = plt.cm.plasma(np.linspace(0,1,len(plot_time_steps)))
    
    plt.clf()
    plt.rcParams.update({'font.size': 20})
    fig, axs = plt.subplots(1,figsize=(8,8),sharex='col')
    fig.tight_layout()
    
    axs.set_xlim(timerange[0],timerange[1])
    axs.set_ylim(np.amin(umax)-0.1,np.amax(umax)+0.1)
    axs.plot(allTimes_1,umax, color ='black')
    axs.set_title('Hight of Soliton/Wave', size=20)
    axs.minorticks_on()
    axs.tick_params(labelsize=20)
    axs.tick_params(direction='in',which='minor',length=5,bottom=True,top=True,left=True,right=True)
    axs.grid()
    axs.set_ylabel('Hight $u_max$')
    
    count = 0
    for item,color in zip(plot_time_steps,colors):

        axs.axvline(x=item, color =color)
        
    axs.set_xlabel('Dimensionless time',fontsize=20)
    plt.tight_layout()
    plt.savefig(str(name)+'.pdf',format='pdf')
    plt.show()
# ----------------------------  Plotting ----------------------------

def IQR(data,time):
        
    q75,q25 = np.percentile(data,[75,25])
    intr_qr = q75-q25
    
    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)
 
    index_list_vel = np.where((data<max)&(data>min))
    
    result_time = [time[i] for i in index_list_vel]
    result_data = [data[i] for i in index_list_vel]
       
    mean_vel = np.mean(result_data)
    std_vel = np.std(result_data)
    
    return result_data,result_time, mean_vel,std_vel


def FILTERED_MEAN_STD(integrals,times,**kwargs):
    
    steps = kwargs.get('steps',0.001)
    filtering = kwargs.get('filtering',True)

    def VALUE_CHUNKER(array, steps):

        maxi = np.amax(array)
        mini = np.amin(array)

        range_list = list(np.arange(mini,maxi+1,steps))

        result = []
        lengths = []

        for i in tqdm(range(len(range_list)-1)):

            lower = range_list[i]
            upper = range_list[i+1]

            sublist = list(array[(array >= lower) & (array<upper)])
            '''
            for j in range(len(array)):

                value = array[j]

                if value >= lower and value<upper:
                    sublist.append(value)
            '''
            result.append(sublist)  
            lengths.append(len(sublist))

        return result, lengths

    if filtering ==True:
        
        q75,q25 = np.percentile(integrals,[75,25])
        intr_qr = q75-q25
        max = q75+(1.5*intr_qr)
        min = q25-(1.5*intr_qr)
        index_list = np.where((integrals<max)&(integrals>min))
        result_data = [integrals[i] for i in index_list][0]

        result, lengths = VALUE_CHUNKER(result_data, steps)
        index = lengths.index(np.max(lengths))

        #mean_value = np.mean(result[index])
        #std_dev = np.std(result[index])

        maxi = np.max(result[index])
        mini = np.min(result[index])
        
    else:
        maxi = np.amax(integrals)
        mini = np.amin(integrals)
    
    filtered_values = []

    for i in range(len(integrals)):

        value = integrals[i]
        time = times[i]

        if value >= mini and value <= maxi:

            filtered_values.append([time, value])
            
    filtered_values = np.array(filtered_values).T
    
    mean_value = np.mean(filtered_values[1])
    std_dev = np.std(filtered_values[1])
    
    return filtered_values, mean_value, std_dev


def VELOCITY_CALC(allData,xrange,allTimes,**kwargs):
    
    useIQR = kwargs.get('useIQR',True)
    
    velocity_list = []
    for i in tqdm(range(0,len(allData)-1)):
        
        step = allData[i]
        time = allTimes[i]
        
        if i ==0:
            step_max = np.amax(step)
            index_step_max = list(step).index(max(list(step)))
            x_step = xrange[index_step_max]

            next_step = allData[i+1]

            next_step_max = np.amax(next_step)
            index_next_step_max = list(next_step).index(max(list(next_step)))
            x_next_step = xrange[index_next_step_max]
            
        else:
            x_step = x_next_step
            
            next_step = allData[i+1]
            next_step_max = np.amax(next_step)
            index_next_step_max = list(next_step).index(max(list(next_step)))
            x_next_step = xrange[index_next_step_max]
            
        v = (x_next_step-x_step)/(allTimes[i+1]-time)
        
        velocity_list.append([time,v])
        
    velocity = np.array(velocity_list).T
    
    if useIQR==True:  
        velocity_data, velocity_time, mean_vel, std_vel = IQR(velocity[1],velocity[0])
    else:
        velocity_data = velocity[1]
        velocity_time = velocity[0]
        mean_vel = np.mean(velocity_data)
        std_vel = np.std(velocity_data)
    
    return [velocity_data, velocity_time, mean_vel, std_vel]


def Plotter_Constants(int_0,int_1,int_2,umax,velocity,allData_1,allTimes_1,integrals_1,xrange,deltaT,name,**kwargs):
    
    timerange = kwargs.get('timerange',[allTimes_1[0],allTimes_1[-1]])
    timeres = kwargs.get('timeres',8000) 
    
    plot_time_steps = [0]
    plot_data_steps = [allData_1[0]]

    for item,data in zip(allTimes_1,allData_1):

        if item-plot_time_steps[-1] >=deltaT*timeres and item>=timerange[0] and item<=timerange[1]:

            plot_time_steps.append(item)
            plot_data_steps.append(data)
        
    plt.rcParams.update({'font.size': 20})
    
    fig, axs = plt.subplots(2,2,figsize=(16,9),sharex='col')
    fig.tight_layout(pad=3)

    formatter = ticker.ScalarFormatter(useMathText=True,useLocale=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1,1))
    
    formatter2 = ticker.ScalarFormatter(useOffset=True,useMathText=True,useLocale=True)
    formatter2.set_scientific(True)
    #formatter2.set_powerlimits((-20,9))
    
    colors = plt.cm.plasma(np.linspace(0,1,len(plot_time_steps)))
    
    axs[0][0].yaxis.set_major_locator(MaxNLocator(integer=True))
    axs[0][0].plot(int_0[0][0],[int_0[1]-int_0[2]]*len(int_0[0][0]),color = 'b',linestyle='dashed')
    axs[0][0].plot(int_0[0][0],[int_0[1]+int_0[2]]*len(int_0[0][0]),color = 'b',linestyle='dashed')

    axs[0][1].plot(int_1[0][0],[int_1[1]-int_1[2]]*len(int_1[0][0]),color = 'b',linestyle='dashed')
    axs[0][1].plot(int_1[0][0],[int_1[1]+int_1[2]]*len(int_1[0][0]),color = 'b',linestyle='dashed')
    
    axs[1][0].plot(int_2[0][0],[int_2[1]-int_2[2]]*len(int_2[0][0]),color = 'b',linestyle='dashed')
    axs[1][0].plot(int_2[0][0],[int_2[1]+int_2[2]]*len(int_2[0][0]),color = 'b',linestyle='dashed')
    
    '''
    axs[0][0].scatter(int_0[0][0],int_0[0][1],color ='black',s=2,marker='*')
    axs[0][1].scatter(int_1[0][0],int_1[0][1],color ='black',s=2,marker='*')
    axs[1][0].scatter(int_2[0][0],int_2[0][1],color ='black',s=2,marker='*')
    '''
    axs[0][0].plot(int_0[0][0],int_0[0][1],marker='o',linestyle='-',markersize=3,color ='black',alpha=0.5)
    axs[0][1].plot(int_1[0][0],int_1[0][1],marker='o',linestyle='-',markersize=3,color ='black',alpha=0.5)
    axs[1][0].plot(int_2[0][0],int_2[0][1],marker='o',linestyle='-',markersize=3,color ='black',alpha=0.5)
    
    axs[0][0].plot(int_0[0][0],[int_0[1]]*len(int_0[0][0]),color ='b')
    axs[0][1].plot(int_1[0][0],[int_1[1]]*len(int_1[0][0]),color ='b')
    axs[1][0].plot(int_2[0][0],[int_2[1]]*len(int_2[0][0]),color ='b')
    
    
    maxi0 = max([int_0[1]+abs(int_0[1]-np.amin(int_0[0][1])),np.amax(int_0[0][1])])
    mini0 = min([int_0[1]-abs(int_0[1]-np.amax(int_0[0][1])),np.amin(int_0[0][1])])
    axs[0][0].set_ylim(mini0,maxi0)
    
    #maxi= max([int_1[1]+int_1[2]+0.1*(int_1[2]),np.amax(int_1[0][1])])
    #mini= min([int_1[1]-int_1[2]-0.1*(int_1[2]),np.amin(int_1[0][1])])
    maxi1 = max([int_1[1]+abs(int_1[1]-np.amin(int_1[0][1])),np.amax(int_1[0][1])])
    mini1 = min([int_1[1]-abs(int_1[1]-np.amax(int_1[0][1])),np.amin(int_1[0][1])])
    axs[0][1].set_ylim(mini1,maxi1)
    
    #maxi= max([int_2[1]+int_2[2]+0.1*(int_2[2]),np.amax(int_2[0][1])])
    #mini= min([int_2[1]-int_2[2]-0.1*(int_2[2]),np.amin(int_2[0][1])])
    maxi2 = max([int_2[1]+abs(int_2[1]-np.amin(int_2[0][1])),np.amax(int_2[0][1])])
    mini2 = min([int_2[1]-abs(int_2[1]-np.amax(int_2[0][1])),np.amin(int_2[0][1])])
    axs[1][0].set_ylim(mini2,maxi2)
    
    axs4 = axs[1][1].twinx()
    #axs4.set_yscale('log')
    
    velocity_data, velocity_time, mean_vel, std_vel = velocity[0],velocity[1],velocity[2],velocity[3]
    #axs4.set_yticks([round(mean_vel-std_vel),round(mean_vel),round(mean_vel+std_vel)])
    axs[0][0].set_yticks([int_0[1]-int_0[2],int_0[1],int_0[1]+int_0[2]])
    #axs[0][1].set_yticks([round(int_1[1]-int_1[2]),round(int_1[1]),round(int_1[1]+int_1[2])])
    #axs[1][1].set_yticks([round(int_2[1]-int_2[2]),round(int_2[1]),round(int_2[1]+int_2[2])])
    
    
    axs4.set_ylabel('Velocity',fontsize=20,color='green')
    axs[1][1].set_ylabel('Wave-height',fontsize=20)
    
    axs[1][1].plot(umax[0],umax[1], color ='black',marker='*',linestyle='-',markersize=3,alpha=0.5)
    axs4.scatter(velocity_time,velocity_data, color = 'green',marker='*',s=3,alpha=0.5)
    
    # Title
    axs[0][0].set_title('Integral $P_0$', size=20,pad=10,loc='right',fontweight='bold')
    axs[0][1].set_title('Integral $P_1$', size=20,pad=10,loc='right',fontweight='bold')
    axs[1][0].set_title('Integral $P_2$', size=20,pad=10,loc='right',fontweight='bold')
    axs[1][1].set_title('Wave-height, Velocity', size=20,pad=10,loc='right',fontweight='bold')
    
    # Tick params
    axs[0][0].minorticks_on()
    axs[0][1].minorticks_on()
    axs[1][0].minorticks_on()
    axs[1][1].minorticks_on()
    
    axs[0][0].tick_params(labelsize=20)
    axs[0][1].tick_params(labelsize=20)
    axs[1][0].tick_params(labelsize=20)
    axs[1][1].tick_params(labelsize=20)
    
    axs[0][0].tick_params(direction='in',which='minor',length=5,bottom=True,top=True,left=True,right=True)
    axs[0][1].tick_params(direction='in',which='minor',length=5,bottom=True,top=True,left=True,right=True)
    axs[1][0].tick_params(direction='in',which='minor',length=5,bottom=True,top=True,left=True,right=True)
    axs[1][1].tick_params(direction='in',which='minor',length=5,bottom=True,top=True,left=True,right=True)
    
    # grid
    #axs[0][0].set_yticks(np.arange(mini0, maxi0, 4),major=True)
    axs[0][0].grid()
    axs[0][1].grid()
    axs[1][0].grid()
    axs[1][1].grid()
    
    # xlabel
    axs[1][0].set_xlabel('Dimensionless time',fontsize=20)
    axs[1][1].set_xlabel('Dimensionless time',fontsize=20)
    
    
    count = 0
    for item,data,color in zip(plot_time_steps,plot_data_steps,colors):

        axs[0][0].axvline(x=item, color =color,alpha=0.5)
        axs[0][1].axvline(x=item, color =color,alpha=0.5)
        axs[1][0].axvline(x=item, color =color,alpha=0.5)
        axs[1][1].axvline(x=item, color =color,alpha=0.5)
    
    current_values = axs[0][0].get_yticks()
    axs[0][0].set_yticklabels(['{:,.0f}'.format(x) for x in current_values])
    
    axs[0][0].yaxis.set_major_formatter(formatter2)
    axs[0][1].yaxis.set_major_formatter(formatter)
    axs[1][0].yaxis.set_major_formatter(formatter)
    
    plt.savefig(str(name)+'_constants.pdf',format='pdf')
    plt.show()
    