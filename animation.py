import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from FVM import vicsekmodel
#from FVMchannel import vicsekmodel
from datetime import datetime


x_length=6
y_length=6
steps=500
bound="channel"
simulation_cls = vicsekmodel(100,timesteps=steps,discard=0,Lx=x_length,Ly=y_length,repeat=1,boundary=bound,save_interval=1, imflag=True, imflag2=True, plotflag=False)
eta=1
radius=1
speed=0.02
pressure_change=0.
diss=1
l0=0.1
attraction=0
flow_align_effect=1
aspect=5
mu=1
force=1

simulation_cls.run_sim(eta=eta, radius=radius, speed=speed, pressure_change=pressure_change, diss=diss, l0=l0, attraction=attraction, flow_align_effect=flow_align_effect, aspect=aspect, mu=mu, force=force)

#%%

#Quiver animation of agents

allpos = simulation_cls.micro_state[0,:,:,0:2]
allvel = simulation_cls.micro_state[0,:,:,2:4]
        
X, Y = allpos[0,:,0],allpos[0,:,1]
U, V = allvel[0,:,0],allvel[0,:,1]
skip=0
fig, ax = plt.subplots(1,1,figsize=(10,10))
plt.close()
Q = ax.quiver(X, Y, U, V, pivot='mid', color='b', units='inches')
ax.axis('scaled')
ax.set_xlim(0,x_length)
ax.set_ylim(0,y_length)
txt=f"$\eta=${eta}, $r=${radius}, $v=${speed}, $\Delta P=${pressure_change}, $\Delta=${diss}, $l_0=${l0}, $\gamma=${attraction}, $\delta=${flow_align_effect}, $a=${aspect}, $\mu=${mu}, $F=${force}"
fig.text(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)

def update_quiver(num, Q, X, Y):
    """updates the horizontal and vertical vector components by a
    fixed increment on each frame
    """
    U, V = allvel[num,:,0],allvel[num,:,1]
    offsets = allpos[num]

    Q.set_offsets(offsets)
    Q.set_UVC(U,V)

    return Q,

# you need to set blit=False, or the first set of arrows never gets
# cleared on subsequent frames
anim = animation.FuncAnimation(fig, update_quiver, fargs=(Q, X, Y),
                               interval=50, frames=steps, blit=False)

rc('animation', html='jshtml')
anim


#%%

#Animation of streamlines

import matplotlib.colors as colors

from google.colab import files

n=simulation_cls.n
x = np.linspace(0, simulation_cls.Lx, n)
y = np.linspace(0, simulation_cls.Ly, n)

fig1, ax1 = plt.subplots(figsize=(10, 10))
txt=f"$\eta=${eta}, $r=${radius}, $v=${speed}, $\Delta P=${pressure_change}, $\Delta=${diss}, $l_0=${l0}, $\gamma=${attraction}, $\delta=${flow_align_effect}, $a=${aspect}, $\mu=${mu}, $F=${force}"
fig1.text(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
#u,v = simulation_cls.stream_plot2(0,force,x,y,n)
#stream = ax1.streamplot(x, y, u, v, color=np.sqrt(u**2+v**2), cmap="binary")
def an(iter):
    ax1.collections = [] # clear lines streamplot
    ax1.patches = [] # clear arrowheads streamplot
    u = simulation_cls.plot_store[0,iter,:,:,0]
    v = simulation_cls.plot_store[0,iter,:,:,1]
    velocity=np.sqrt(u**2+v**2)
    stream = ax1.streamplot(x, y, u, v, color=velocity, norm=colors.Normalize(vmin=0, vmax=0.1*simulation_cls.posminlxly), cmap="binary", zorder=0)
    #stream = ax1.streamplot(x, y, u, v, color="black", zorder=0)
    ax1.quiver(simulation_cls.micro_state[0,iter,:,0], simulation_cls.micro_state[0,iter,:,1], simulation_cls.micro_state[0,iter,:,2], simulation_cls.micro_state[0,iter,:,3], width=0.01*simulation_cls.Lx, color="red", zorder=10)
    ax1.grid(False)
    ax.set_xlim(0,simulation_cls.Lx)
    ax.set_ylim(0,simulation_cls.Ly)
    print(iter)
    return stream

animo = animation.FuncAnimation(fig1, an, frames=200, interval=100, blit=False, repeat=False)

animo.save(f'stream-{bound}-s{speed}-f{force}.gif', writer=animation.PillowWriter(fps=60))
files.download(f'stream-{bound}-s{speed}-f{force}.gif')

