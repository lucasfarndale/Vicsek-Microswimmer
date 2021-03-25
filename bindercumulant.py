import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from VM import vicsekmodel
from datetime import datetime

n=10
reps=10
opvals = np.zeros(n)

ls = np.linspace(1,25,n)
es = np.linspace(0.18,0.20,n)

binder=np.zeros(n)

for j in range(n):
    print(j)
    simulation_cls = vicsekmodel(5000,timesteps=500,discard=0,L=5,repeat=reps, boundary="periodic", save_interval=1)
    simulation_cls.run_sim(radius=1, eta=es[j], speed=0.03, l0=0.1)
    op=simulation_cls.macro_state1
    avop=np.mean(op[:,100:], axis=1)
    
    av2=np.mean(avop**2, axis=0)
    av4=np.mean(avop**4, axis=0)
    av22=av2**2

    binder[j]=1-av4/(3*av22)

plt.plot(es, binder)