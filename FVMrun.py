import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from FVM import vicsekmodel
#from FVMchannel import vicsekmodel
from datetime import datetime




n_length=50
n_eta=50
n_force=50
opvals = np.zeros((n_length,n_eta))
nevals = np.zeros((n_length,n_eta))
tpvals = np.zeros((n_length,n_eta))

steps=500
bound="square"
radius=1
speed=0.02
pressure_change=0.
diss=1
l0=0.1
attraction=0
flow_align_effect=1
aspect=5
mu=1
force=-1

imflag=True
imflag2=True
plotflag=False

ls = np.linspace(1,25,n_length)
es = np.linspace(0,2*np.pi,n_eta)
fs = -np.logspace(-7, -2, n_force)

#%%

#Length v noise colorplot

n_length=25
n_eta=25
n_force=25
opvals = np.zeros((n_eta,n_length))
nevals = np.zeros((n_eta,n_length))
tpvals = np.zeros((n_eta,n_length))

ls = np.linspace(1,25,n_length)
es = np.linspace(0,2*np.pi,n_eta)
fs = -np.logspace(-4, 2, n_force)

now = datetime.now()
start_time = now.strftime("%H:%M:%S")
print("Start Time =", start_time)

for i in range(n_length):
    for j in range(n_eta):
      print("Step",i,j)
      simulation_cls = vicsekmodel(100,timesteps=steps,discard=0,Lx=ls[i],Ly=ls[i],repeat=1,boundary=bound,save_interval=1, imflag=imflag, imflag2=imflag2, plotflag=plotflag)
      simulation_cls.run_sim(eta=es[j], radius=radius, speed=speed, pressure_change=pressure_change, diss=diss, l0=l0, attraction=attraction, flow_align_effect=flow_align_effect, aspect=aspect, mu=mu, force=force)
      op, ne, _ = simulation_cls.get_macro_states()
      opvals[j,i] = np.mean(op[200:])
      nevals[j,i] = np.mean(ne[200:])
      #tpvals[j,i] = abs(np.mean(tp[200:]))*simulation_cls.Lx/simulation_cls.N

now = datetime.now()
end_time = now.strftime("%H:%M:%S")
print("Start Time =", start_time, "End Time =", end_time)
print("Length v Noise")

#%%
      

fig, ax = plt.subplots()
cmap = ax.pcolormesh(ls, es, opvals, cmap="hot")
fig.colorbar(cmap, label="Order Parameter $\psi$")
#fig.colorbar(cmap, label="Average Local Density $\mathcal{M}$")
#fig.colorbar(cmap, label="Transport Parameter $T$")
ax.set_xlabel("Length $L$")
ax.set_ylabel("Noise $\eta$")

#%%

#Force v noise colorplot

Lx=5
Ly=5

n_length=50
n_eta=50
n_force=50
opvals = np.zeros((n_force,n_eta))
nevals = np.zeros((n_force,n_eta))
tpvals = np.zeros((n_force,n_eta))

ls = np.linspace(1,25,n_length)
es = np.linspace(0,2*np.pi,n_eta)
fs = -np.logspace(-4, 2, n_force)

now = datetime.now()
start_time = now.strftime("%H:%M:%S")
print("Start Time =", start_time)

for i in range(n_force):
    for j in range(n_eta):
      print("Step",i,j)
      simulation_cls = vicsekmodel(100,timesteps=steps,discard=0,Lx=Lx,Ly=Ly,repeat=1,boundary=bound,save_interval=1, imflag=imflag, imflag2=imflag2, plotflag=plotflag)
      simulation_cls.run_sim(eta=es[j], radius=radius, speed=speed, pressure_change=pressure_change, diss=diss, l0=l0, attraction=attraction, flow_align_effect=flow_align_effect, aspect=aspect, mu=mu, force=fs[i])
      op, ne, _, tp = simulation_cls.get_macro_states()
      opvals[j,i] = np.mean(op[300:])
      nevals[j,i] = np.mean(ne[300:])
      tpvals[j,i] = abs(np.mean(tp[300:]))*simulation_cls.Lx/simulation_cls.N
      
now = datetime.now()
end_time = now.strftime("%H:%M:%S")
print("Start Time =", start_time, "End Time =", end_time)
print("Force v Noise")


#%%


fig, ax = plt.subplots()
cmap = ax.pcolormesh(-fs, es, tpvals, norm=colors.LogNorm(), cmap="hot")
#fig.colorbar(cmap, label="Order Parameter $\psi$")
#fig.colorbar(cmap, label="Average Local Density $\mathcal{M}$")
fig.colorbar(cmap, label="Transport Parameter $T$")
ax.set_xscale("log")
ax.set_xlabel("Force $f$")
ax.set_ylabel("Noise $\eta$")

#%%

#Force v length colorplot

eta=1

n_length=25
n_eta=25
n_force=25
opvals = np.zeros((n_length,n_eta))
nevals = np.zeros((n_length,n_eta))
tpvals = np.zeros((n_length,n_eta))

ls = np.linspace(1,25,n_length)
es = np.linspace(0,2*np.pi,n_eta)
fs = -np.logspace(-3, 2, n_force)

now = datetime.now()
start_time = now.strftime("%H:%M:%S")
print("Start Time =", start_time)

for i in range(n_force):
    for j in range(n_length):
      print("Step",i,j)
      simulation_cls = vicsekmodel(100,timesteps=steps,discard=0,Lx=ls[j],Ly=ls[j],repeat=1,boundary=bound,save_interval=1, imflag=imflag, imflag2=imflag2, plotflag=plotflag)
      simulation_cls.run_sim(eta=eta, radius=radius, speed=speed, pressure_change=pressure_change, diss=diss, l0=l0, attraction=attraction, flow_align_effect=flow_align_effect, aspect=aspect, mu=mu, force=fs[i])
      op, ne, _, tp = simulation_cls.get_macro_states()
      opvals[j,i] = np.mean(op[200:])
      nevals[j,i] = np.mean(ne[200:])
      tpvals[j,i] = abs(np.mean(tp[200:]))*simulation_cls.Lx/simulation_cls.N
      
now = datetime.now()
end_time = now.strftime("%H:%M:%S")
print("Start Time =", start_time, "End Time =", end_time)
print("Force v Length")
      
#%%


fig, ax = plt.subplots()
cmap = ax.pcolormesh(-fs, ls, tpvals, cmap="hot")
#fig.colorbar(cmap, label="Order Parameter $\psi$")
#fig.colorbar(cmap, label="Average Local Density $\mathcal{M}$")
fig.colorbar(cmap, label="Transport Parameter $T$")
ax.set_xscale("log")
ax.set_xlabel("Force $f$")
ax.set_ylabel("Length $L$")