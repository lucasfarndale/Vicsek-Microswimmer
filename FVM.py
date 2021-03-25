import numpy as np
import sympy as sp
from math import *
import sys
import random
import time

import matplotlib.pyplot as plt
plt.style.use('ggplot') 
plt.style.use('seaborn-poster') 
plt.style.use('seaborn-whitegrid') 

from IPython import display

from matplotlib import animation, rc
from IPython.display import HTML

from tqdm import tqdm

class vicsekmodel:
    def __init__(self, N, timesteps, discard, repeat, Lx, Ly, save_interval, boundary, imflag=False, imflag2=False, plotflag=False, disable_progress=False):
        self.N = N #N agents
        self.timesteps = timesteps
        self.discard = discard #discards this many frames - saves computing time if wanting to see effects later on
        self.B = repeat  # repeat for B batches
        self.Lx = Lx #x boundary
        self.Ly = Ly #y boundary
        self.posminlxly = 0.1*min(Lx, Ly)
        self.negminlxly = -0.1*min(Lx, Ly)
        self.save_interval = save_interval #interval between frames saved
        self.macro_state1 = np.zeros((self.B, self.timesteps - self.discard)) #order parameter
        self.macro_state2 = np.zeros((self.B, self.timesteps - self.discard)) #counts neighbours in radius
        if self.save_interval>0: 
            self.micro_state = np.zeros((self.B, (self.timesteps - self.discard)//self.save_interval, N, 5))
        self.imflag = imflag
        self.imflag2 = imflag2
        self.plotflag = plotflag
        self.disable_progress = disable_progress #turn off progress bar
        self.boundary_conditions = boundary #string to determine boundary conditions: "periodic", "channel", "square"
        if self.imflag==True or self.plotflag==True:
            self.n = 20
        if self.plotflag == True:
            self.plot_store = np.zeros((self.B, self.timesteps - self.discard, self.n, self.n, 2))


    def initialise_state(self):
        set_initial_positions=False
        if set_initial_positions==True:
            posx=np.array([[[0.4],[0.6]]])
            posy=np.array([[[0.2],[0.25]]])
            self.angles=np.array([[[pi/2],[pi/2]]])
        else:
            posx = np.random.uniform(0, self.Lx, (self.B,self.N,1)) #randomises x positions
            posy = np.random.uniform(0, self.Ly, (self.B,self.N,1)) #randomises y positions
            self.angles = np.random.uniform(0, 2*pi, (self.B,self.N,1)) #randomises orientations

        self.positions = np.concatenate([posx,posy],axis=2) #creates position vector

    

    def macro(self, xx, yy, mu, F):
        zz = xx-yy
        norm = np.linalg.norm(zz)
        G=np.identity(2)/norm + np.tensordot(zz,zz,axes=0)/norm**3
        dot = np.matmul(G, F)

        return 1/(8*np.pi*mu)*dot

    def macro2(self, xx, XX, mu, F):
        flow=0
        count=-1
        for yy in XX:
            count+=1
            if ((yy==xx).all())==False:
                flow+=self.macro(xx, yy, mu, F[count,:])
        return flow

    
    def stressmacro_fast(self, xx, yy, mu, cosA, sinA):
        z = yy-xx
        d=np.linalg.norm(z)
        flow=z*(-z[1]**2-z[0]**2+3*(z[0]*cosA+z[1]*sinA)**2)/d**5
        l=0.01
        return flow

    def stressmacro2(self, xx, XX, mu, cosA, sinA):
        flow=0
        count=-1
        for yy in XX:
            count+=1
            if ((yy==xx).all())==False:
                flow+=self.stressmacro_fast(xx, yy, mu, cosA[count,0], sinA[count,0])
        return flow


    def run_sim(self, eta, radius, speed, pressure_change, diss, l0, attraction, flow_align_effect, aspect, mu, force):

        if self.imflag == True:
            if self.imflag2 == True:
                mat = np.tile(np.eye(3*self.N), self.B)
            else:
                mat = np.tile(np.eye(2*self.N), self.B)
        else:
            mat = np.tile(np.eye(self.N), self.B)


        def stresslet_fast(G, distances, cosA, sinA):
            Gx=G[:,:,:,0:1]
            Gy=G[:,:,:,1:2]
            T=G*(-Gx**2-Gy**2+3*(Gx*cosA+Gy*sinA)**2)/distances[:,:,:,np.newaxis]**5

            T=np.where(np.isnan(T), 0, T)

            flow=np.sum(T, axis=2)
            
            l=0.01

            return l*force/(8*np.pi*mu)*flow

        def stlt(distance_vectors, distances, Fmatrix):
            G = mat[...,np.newaxis, np.newaxis]/distances[...,np.newaxis, np.newaxis] + np.einsum('bijk, bijl -> bijkl', distance_vectors, distance_vectors)/distances[:,:,:,np.newaxis,np.newaxis]**3
            G = np.where(np.isnan(G), 0, G)
            dot = np.einsum('bijkl, bim -> bim', G, Fmatrix)
            return 1/(8*np.pi*mu)*dot
        
        def dudx(distance_vectors, dist, Fmatrix):
            grads = np.zeros((self.B, self.N, 4, 1))
            pm=8*np.pi*mu
            for b in range(self.B):
                for i in range(self.N):
                    for j in range(self.N):
                        if i!=j:
                            x1=distance_vectors[b,i,j,0]
                            x2=distance_vectors[b,i,j,1]
                            x12=x1**2
                            x22=x2**2
                            dm12=(dist[b,i,j]-3*x12)
                            dm22=(dist[b,i,j]-3*x22)
                            ndm12=(-dist[b,i,j]-3*x12)
                            ndm22=(-dist[b,i,j]-3*x22)
                            f1x1=Fmatrix[b,i,0]*x1
                            f1x2=Fmatrix[b,i,0]*x2
                            f2x1=Fmatrix[b,i,1]*x1
                            f2x2=Fmatrix[b,i,1]*x2
                            con=pm*dist[b,i,j]**5
                            a=(f1x1+f2x2)
                            grads[b,i,0,:]+=a*dm12
                            grads[b,i,3,:]+=a*dm22
                            grads[b,i,1,:]+=(f1x2*ndm12+f2x1*dm22)
                            grads[b,i,2,:]+=(f1x2*dm12+f2x1*ndm22)
                            grads[b,i,:,:]=grads[b,i,:,:]/con
            return grads
        

        
        def stressdudx_fast(distance_vectors, dist, cosA, sinA):            
            grads = np.zeros((self.B, self.N, self.N, 4))
                            
            x, y = distance_vectors[...,0], distance_vectors[...,1]

            x2, x3, x4, y2, y4, dd7, cos2A, sin2A = x**2, x**3, x**4, y**2, y**4, 2*dist[:,:,:]**7, 2*cosA**2-1, 2*sinA*cosA

            grads[:,:,:,0]=6*x*y*sin2A*(2*y2-3*x2)-3*cos2A*(2*x4-7*x2*y2+y4)-2*x4-x2*y2+y4
            grads[:,:,:,1]=3*x*(cos2A*(7*x*y2-3*x3)+2*x*sin2A*(x2-4*y2)-y*(x2+y2))
            grads[:,:,:,2]=3*y*(cos2A*(7*x*y2-3*x3)+2*y*sin2A*(y2-4*x2)-x*(x2+y2))
            grads[:,:,:,3]=6*x*y*sin2A*(2*x2-3*y2)+3*cos2A*(x4-7*x2*y2+2*y4)+(x2-2*y2)*(x2+y2)

            grads=grads/(2*dd7[...,np.newaxis])

            grads=np.where(np.isnan(grads), 0, grads)
            
            final=np.sum(grads, axis=2)

            l=0.01
            return l*force*np.expand_dims(final,-1)/(8*np.pi*mu)
        
        def plotfunc(pos, f, x, y, n):
            vels = np.zeros((self.B, n, n, 2))
            for b in range(self.B):
                count1=-1
                for i in x:
                    count2=-1
                    count1+=1
                    for j in y:
                        count2+=1
                        ij = np.array([i,j])
                        vels[b,count2,count1,:]=self.macro2(ij, pos[b,:,:], mu, f[b,:,:])
                        #if j==0:
                            #print(vels[b,count2,count1,:])
            u=vels[0,:,:,0]
            v=vels[0,:,:,1]
            return u, v

        def plotfuncstress(pos, x, y, n, cosA, sinA, force):
            vels = np.zeros((self.B, n, n, 2))
            for b in range(self.B):
                count1=-1
                for i in x:
                    count2=-1
                    count1+=1
                    for j in y:
                        count2+=1
                        ij = np.array([i,j])
                        vels[b,count2,count1,:]=self.stressmacro2(ij, pos[b,:,:], mu, cosA[b,:,:], sinA[b,:,:])
                        #if j==0:
                            #print(vels[b,count2,count1,:])
            l=0.01
            u=l*force*vels[0,:,:,0]/(8*np.pi*mu)
            v=l*force*vels[0,:,:,1]/(8*np.pi*mu)
            return u, v




        def update(X, A):

            cos_A = np.cos(A)
            sin_A = np.sin(A)


            if self.imflag == True:    
                A2 = -A
                cos_A2 = np.cos(A2)
                sin_A2 = np.sin(A2)
                if self.imflag2 == True:
                    A3 = -A
                    cos_A3 = np.cos(A3)
                    sin_A3 = np.sin(A3)
            
            if self.plotflag == True:
                plotx = np.linspace(0, self.Lx, self.n)
                ploty = np.linspace(0, self.Ly, self.n)
            
            
            neigh = 0

            #Calculates arrays of scalar and vector distances between agents
            Xcoords = np.expand_dims(X[...,0],-1)
            dx = -Xcoords + np.matrix.transpose(Xcoords, [0,2,1])

            Ycoords = np.expand_dims(X[...,1],-1)
            dy = -Ycoords + np.matrix.transpose(Ycoords, [0,2,1])

            if self.boundary_conditions == "periodic":
                dx = np.where(dx>0.5*self.Lx, dx-self.Lx, dx)
                dx = np.where(dx<-0.5*self.Lx, dx+self.Lx, dx)
            
                dy = np.where(dy>0.5*self.Ly, dy-self.Ly, dy)
                dy = np.where(dy<-0.5*self.Ly, dy+self.Ly, dy)
            
            elif self.boundary_conditions == "channel":
                dx = np.where(dx>0.5*self.Lx, dx-self.Lx, dx)
                dx = np.where(dx<-0.5*self.Lx, dx+self.Lx, dx)

            dist_vectors = np.concatenate([np.expand_dims(dx, -1), np.expand_dims(dy, -1)], axis=-1) #distance vectors

            dist = np.sqrt(np.square(dx)+np.square(dy))#distances

            #finds neighbours within interaction radius, and creates arrays of their x and y orientation coordinates respectively
            neighbours_x = np.where(dist<=radius, cos_A, 0)
            neighbours_y = np.where(dist<=radius, sin_A, 0)

            #finds average number of neighbours for each agent
            neigh = np.count_nonzero(np.add(neighbours_x,neighbours_y))/self.N

            #aligns agents to neighbours
            #align_x = np.where(dist<=radius, cos_A, np.zeros_like(cos_A))
            align_x = np.mean(neighbours_x,axis=1)

            #align_y = np.where(dist<=radius, sin_A, np.zeros_like(sin_A))
            align_y = np.mean(neighbours_y,axis=1)

            #creates array of angles and adds noise
            A = np.expand_dims(np.arctan2(align_y, align_x),-1) + np.random.uniform(low=-eta/2, high=eta/2, size=(self.B,self.N,1))

            #attractive/repulsive forces
            if attraction != 0:
                shiftx = np.where(dist<=1, attraction*(dist-l0)*dx, 0)
                avx = np.mean(shiftx, axis=1)
                avx = np.expand_dims(avx, -1)
                
                shifty = np.where(dist<=1, attraction*(dist-l0)*dy, 0)
                avy = np.mean(shifty, axis=1)
                avy = np.expand_dims(avy, -1)
                
                shift = np.concatenate([avx, avy], axis=2)
            else:
                shift = 0

            #stokes_force = -force*np.concatenate([cos_A,sin_A],axis=-1)
            #stokesflow = np.zeros((self.B, self.N, 2))

            #dUdX = dudx(dist_vectors, dist, stokes_force)
            dUdX = stressdudx_fast(dist_vectors, dist, cos_A, sin_A)

            
            """
            Method of  images
            """
            if self.imflag == True:
                if self.imflag2 == True:
                    Xcoords = np.concatenate((Xcoords, Xcoords, Xcoords),axis=1)
                    dx = -Xcoords + np.matrix.transpose(Xcoords, [0,2,1])
                else:
                    Xcoords = np.concatenate((Xcoords, Xcoords),axis=1)
                    dx = -Xcoords + np.matrix.transpose(Xcoords, [0,2,1])
                
                if self.imflag2 == True:
                    Ycoords = np.concatenate((Ycoords, -Ycoords, 2*self.Ly-Ycoords),axis=1)
                    dy = -Ycoords + np.matrix.transpose(Ycoords, [0,2,1])
                else:
                    Ycoords = np.concatenate((Ycoords, -Ycoords),axis=1)
                    dy = -Ycoords + np.matrix.transpose(Ycoords, [0,2,1])

                dist_vectors = np.concatenate([np.expand_dims(dx, -1), np.expand_dims(dy, -1)], axis=-1) #distance vectors
            
                dist = np.sqrt(np.square(dx)+np.square(dy))#distances
  
                #stokes_force2 = -force*np.concatenate([cos_A2, sin_A2], axis=-1)
                #stokes_force = np.concatenate((stokes_force,stokes_force2), axis=1)

                if self.imflag2 == True:
                    cos_A=np.concatenate([cos_A, cos_A2, cos_A3],axis=1)
                    sin_A=np.concatenate([sin_A, sin_A2, sin_A3],axis=1)
                else:
                    cos_A=np.concatenate([cos_A, cos_A2],axis=1)
                    sin_A=np.concatenate([sin_A, sin_A2],axis=1)

                
            if self.plotflag == True:
                pos = np.concatenate([Xcoords,Ycoords], axis=-1)
                #u, v = plotfunc(pos, stokes_force, plotx, ploty, self.n)
                u, v = plotfuncstress(pos, plotx, ploty, self.n, cos_A, sin_A, force)
                self.plot_store[0,counter,:,:,0] = u
                self.plot_store[0,counter,:,:,1] = v
                
            
            #stokesflow = stlt(dist_vectors, dist, stokes_force)
            stokesflow = stresslet_fast(dist_vectors, dist, cos_A, sin_A)

            stokesflow = np.where(stokesflow>self.posminlxly, self.posminlxly, stokesflow)
            stokesflow = np.where(stokesflow<self.negminlxly, self.negminlxly, stokesflow)

            

            sin_A=np.sin(A)
            cos_A=np.cos(A)

            A += (1/(aspect**2+1))*(0.5*(aspect**2-1)*np.sin(2*A)*(dUdX[...,3,:]-dUdX[...,0,:])+((aspect**2)*(cos_A**2)+(sin_A**2))*dUdX[...,2,:]-((aspect**2)*(sin_A**2)+(cos_A**2))*dUdX[...,1,:])

            #velocity in orientation diretion
            velocity = speed*np.concatenate([np.cos(A),np.sin(A)],axis=-1)

            #calculate new positions
            X += velocity + shift

            flow = pressure_change*X[...,1]*(self.Ly-X[...,1])
            flowdiff = pressure_change*(self.Ly-2*X[...,1])
            flowdiff = np.expand_dims(flowdiff, -1)

            X[...,0] += flow
            


            A += (1/((aspect**2)+1))*((aspect**2)*(np.sin(A)**2)-(np.cos(A)**2))*flowdiff


            X += stokesflow[:,0:self.N,:]
            

            if self.boundary_conditions == "periodic":
                A = np.where(A<-pi,  A%2*pi, A)
                A = np.where(A>pi, A%2*pi, A)

                X[...,0:1] = np.where(X[...,0:1]>self.Lx, X[...,0:1]%self.Lx, X[...,0:1])
                X[...,1:2] = np.where(X[...,1:2]>self.Ly, X[...,1:2]%self.Ly, X[...,1:2])
                X[...,0:1] = np.where(X[...,0:1]<0, X[...,0:1]%self.Lx, X[...,0:1])
                X[...,1:2] = np.where(X[...,1:2]<0, X[...,1:2]%self.Ly, X[...,1:2])
                
            elif self.boundary_conditions == "channel":
                A = np.where(X[...,1:2]>self.Ly, -A, A)
                A = np.where(X[...,1:2]<0, -A, A)
                
                A = np.where(A<-pi,  A%2*pi, A)
                A = np.where(A>pi, A%2*pi, A)
                
                X[...,0:1] = np.where(X[...,0:1]>self.Lx, X[...,0:1]%self.Lx, X[...,0:1])
                X[...,0:1] = np.where(X[...,0:1]<0, X[...,0:1]%self.Lx, X[...,0:1])

                X[...,1:2] = np.where(X[...,1:2]>self.Ly, self.Ly-diss*(X[...,1:2]%self.Ly), X[...,1:2])
                X[...,1:2] = np.where(X[...,1:2]<0, -diss*X[...,1:2]%self.Ly, X[...,1:2])

            elif self.boundary_conditions == "square":
                
                A = np.where(X[...,0:1]>self.Lx, pi-A, A)
                A = np.where(X[...,0:1]<0, pi-A, A)

                A = np.where(X[...,1:2]>self.Ly, -A, A)
                A = np.where(X[...,1:2]<0, -A, A)
                
                A = np.where(A<-pi,  A%2*pi, A)
                A = np.where(A>pi, A%2*pi, A)

                X[...,0:1] = np.where(X[...,0:1]>self.Lx, self.Lx-diss*(X[...,0:1]%self.Lx), X[...,0:1])
                X[...,1:2] = np.where(X[...,1:2]>self.Ly, self.Ly-diss*(X[...,1:2]%self.Ly), X[...,1:2])
                X[...,0:1] = np.where(X[...,0:1]<0, -diss*X[...,0:1]%self.Lx, X[...,0:1])
                X[...,1:2] = np.where(X[...,1:2]<0, -diss*X[...,1:2]%self.Ly, X[...,1:2])

            else:
                print("Boundary conditions not understood")
            if np.any(X>max(self.Lx, self.Ly)):
                print("Agent outside boundary")     

            return X, A, neigh
            
        self.initialise_state()

        counter=0
        for i in tqdm(range(self.timesteps),disable=self.disable_progress, position=0, leave=True):
            self.positions, self.angles, self.tot_neigh = update(self.positions,  self.angles)
            if i>=self.discard:
                self.macro_state1[:,i-self.discard] = self.compute_macro_state1()
                self.macro_state2[:,i-self.discard] = self.compute_macro_state2()
                if self.save_interval>0: 
                    if i%self.save_interval==0:
                        self.micro_state[:,counter,:,0:2] = self.positions
                        self.micro_state[:,counter,:,2:3] = np.cos(self.angles)
                        self.micro_state[:,counter,:,3:4] = np.sin(self.angles)
                        self.micro_state[:,counter,:,4:5] = self.tot_neigh*np.ones(np.shape(self.angles))  
                        counter = counter + 1

        return 

    def compute_macro_state1(self):
        # return the order parameter for each batch

        A = self.angles
        cos_A = np.cos(A)
        sin_A = np.sin(A)

        velocity = np.concatenate([cos_A,sin_A],axis=-1)

        av_velocity = np.mean(velocity,axis=1)

        order_parameter = np.linalg.norm(av_velocity,axis=1)
        
        return order_parameter
    
    def compute_macro_state2(self):
 
        neighbours_correlation = self.tot_neigh
        
        return neighbours_correlation
        


    def stream_plot2(self, i, force, x, y, n):
        pos = self.micro_state[:,i,:,0:2]
        f=-force*np.stack([self.micro_state[:,i,:,2],self.micro_state[:,i,:,3]],axis=-1)
        vels = np.zeros((self.B, n, n, 2))
        for b in range(self.B):
            count1=-1
            for i in x:
                count2=-1
                count1+=1
                for j in y:
                    count2+=1
                    ij = np.array([i,j])
                    vels[b,count2,count1,:]=self.macro2(ij, pos[b,:,:], mu, f[b,:,:])
                    #if j==0:
                    #    print(vels[b,count2,count1,:])
        u=vels[0,:,:,0]
        v=vels[0,:,:,1]
        return u, v
        

                    
            




    def get_macro_states(self):
    
        macro_x = np.ravel(self.macro_state1[:,:-1])
        macro_dx = np.ravel(self.macro_state2[:,:-1]) #now the total number of neighbours within a radius of 1.   np.ravel(np.diff(self.macro_state))
        macro_g = 0.5*(macro_dx**2)     #tot_neigh  
        #tot_neigh    #0.5*(macro_dx**2)
        
        return macro_x, macro_dx, macro_g