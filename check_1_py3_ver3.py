# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 09:08:30 2019

@author: leeva
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import jit
from progressbar import Percentage, ProgressBar,Bar,ETA


"""
Initialise function and nearest neigbours boundary conditions check function

"""
def init(n):
    down = -1
    
    i_matrix = np.random.randint(0,1+1,size=(n,n))
    
    for i in range(n):
        for j in range(n):
            if i_matrix[i][j] == 0:
                i_matrix[i][j] = down
    return i_matrix

@jit(nopython = True)
def nnb(config,i,j):
        i_matrix = config
        up = n-1

        if i == 0: i1 = up
        else:  i1 = i - 1

        if i == up: i2 = 0
        else:  i2 = i + 1
    
        if j == up:  j1 = 0
        else: j1 = j+1
    
        if j == 0: j2 = up
        else: j2 = j - 1
    
        nb1 = np.array([i_matrix[i2,j],i_matrix[i,j1],i_matrix[i1,j],i_matrix[i,j2]])

        #nns =  [nn1,nn2,nn3,nn4]
        #nns = np.sum(nns)
        return nb1

"""
mc and mc_k functions each performing glauber and kawaski dynamics repectively
"""    
@jit(nopython = True)#,parallel = True)
def mc(config,te,mc_step,n):
        #print('g')
        if dynamic == 1:
            #print('here')
            for i in range(mc_step):
                cost = 0
                a = int(np.random.uniform(0, n))
                b = int(np.random.uniform(0, n))
                s =  config[a, b]
#                nb = config[(a+1)%n,b] + config[a,(b+1)%n] + config[(a-1)%n,b]
#                + config[a,(b-1)%n]
                cost = 2*s*np.sum(nnb(config,a,b))
                if cost <= 0:
                    s *= -1
                elif np.random.rand() < np.exp(-(1*cost)/te):
                    s *= -1
                config[a, b] = s
        return config
    
@jit(nopython = True)
def mc_k(config,te,mc_step,n):
        #print('k')
        for i in range(mc_step):
            #x = 1
        #while x == 1:
            a1 = int(np.random.randint(0, n))
            b1 = int(np.random.randint(0, n))
            a2 = int(np.random.randint(0, n))
            b2 = int(np.random.randint(0, n))
            #a2 = (a1+1)%n
            #b2 = b1
            s1 =  config[a1,b1]
            s2 =  config[a2,b2]
            
            if s1 != s2:
                #x = 0
                    
            
                #nb1 = config[(a1+1)%(n-1),b1] + config[a1,(b1+1)%(n-1)] 
                #+ config[(a1-1)%(n-1),b1] + config[a1,(b1-1)%(n-1)]
                #nb2 = config[(a2+1)%(n-1),b2] + config[a2,(b2+1)%(n-1)] 
                #+ config[(a2-1)%(n-1),b2] + config[a2,(b2-1)%(n-1)]          
                
                #nb = nnb(config,a1,a2)
                
                #cost1 = 2*s1*nnb(config,a1,a2)
                #cost2 = 2*s2*nns
                #cost =  s1*(nb1-nb2) 
                cost = 2*s1*np.sum(nnb(config,a1,b1)) + 2*s2*np.sum(nnb(config,a2,b2))
                #print (cost)
                
                if a1+1==a2 and b1==b2: cost+=4
                elif a1-1==a2 and b1==b2: cost+=4
                elif a1==a2 and b1+1==b2: cost+=4
                elif a1==a2 and b1-1==b2: cost+=4
                
                
                if cost <= 0:
                    config[a1, b1] = s2
                    config[a2, b2] = s1
                elif np.random.rand() < np.exp(-(1*cost)/te):
                    config[a1, b1] = s2
                    config[a2, b2] = s1
            #k += 1
        return config
"""
Measurement functions, taking lists of data to generate means and required 
measurements
""" 
@jit(nopython = True)   
def Measure_Energy(config,n):
    energy = 0
    for i in range(len(config)):
        for j in range(len(config)):
            S = config[i,j]
            nb = nnb(config,i,j)#config[(i+1)%n-1, j] + config[i,(j+1)%n-1] + config[(i-1)%n-1, j]+ config[i,(j-1)%n-1]
            energy += -1*S*(nb[1]+nb[2])
    e2 = energy**2
    #print(energy/4)
    return energy,e2

@jit(nopython = True)
def Susceptibility(a1,a2,te,N):    
    avg_m = np.mean(a1)
    avg_m2 = np.mean(a2)
    X = (avg_m2 - avg_m*avg_m)/(te*N)
    return avg_m, X

@jit(nopython = True)
def Measure_Mag(config,sq):
    if sq == False:
        m = np.sum(config)
    elif sq == True:
        m = (np.sum(config))**2
    return np.abs(m)
"""
Error calcuculation functions taking in the relavant measurements as arrays

sigma_sX performs the bootstrapping error determination with number of samples
set to >= 1000 samples to get the best representation of the expected error.
"""
@jit(nopython = True)
def sigma_sX(sx,num,samp):
    ar = np.zeros(num-1)
    sam = np.zeros(samp)
    for k in range(samp):
        for j in range(len(ar)):
            x = np.random.randint(0,high=num)
            ar[j] = sx[x]
        avg = np.mean(ar)
        sam[k] = avg
    cmean = np.mean(sam)**2
    c2mean = np.mean(sam*sam)
    erc = np.sqrt((c2mean-cmean))
    return erc

@jit(nopython = True)
def sigma_m(m,num):

    er1 = np.std(m)/(len(m))**0.5
    return er1

def updatefig(*args):
    if dynamic == 1:
        im.set_data(mc(config,temp,100,n))
    elif dynamic == 2:
        im.set_data(mc_k(config,temp,1000,n))        
    return im,



print('input system size:')
n = int(input())
print('input dynamics: 1 == Glauber, 2 == Kawasaki:')
dynamic = int(input())
print('run visulisation? 1==y ,0==n')
a = int(input())
print('input sim tempL')
t = float(input())




#n = 50
N = n*n
#dynamic = 2
#a = 0


n_mes = 400

#Lists init
E_list = []
M_list = []
Cv_1 = []
Cv_sigma = []
X_1 = []
X_sigma = []
M_sigma = []
E_sigma = []

#data mesurement range set-up
nt = 30
passes = 2000
mc_step = 1000
T =  np.linspace(1.2,3,nt)

bar = ProgressBar(widgets=[Bar('=', '[', ']'), ' ', Percentage(), ' ', ETA(),'\n'], maxval=nt).start()
##################__Main loop_##################
if a != 1:
    z = 0
    config = init(n)
    for i in T:
        temp = i
        sweeps = 0
        m_list = []
        m2_list = []
        e_list = []
        e2_list = []
        Cv_list = np.zeros(n_mes)
        e_Avg = np.zeros(n_mes)
        m_avg = np.zeros(n_mes)
        X_avg = np.zeros(n_mes)
        
        for k in range(n_mes):
            for j in range(passes):
                if dynamic == 1:
                    #print('g')
                    mc(config,temp,mc_step,n)
                elif dynamic == 2:
                    mc_k(config,temp,mc_step,n)
                    
                #Equlibrium condition and autocorrelation aviodance condition
                    
                if sweeps >= 200:
                    if sweeps%10 == 0:
                        m_list.append(Measure_Mag(config,sq=False))
                        m2_list.append(Measure_Mag(config,sq=True))
                        e,e2 = Measure_Energy(config,n)
                        e_list.append(e)
                        e2_list.append(e2)
                sweeps += 1
            
            e1 = np.asarray(e_list)
            e2 = np.asarray(e2_list)
            m1 = np.asarray(m_list)
            m2 = np.asarray(m2_list)
            avg_E, CvN = Susceptibility(e1,e2,temp,N)
            avg_mag, X = Susceptibility(m1,m2,temp,N)
            
            e_Avg[k] = avg_E/N
            Cv_list[k] = (CvN/temp)
            X_avg[k] = (X)
            m_avg[k] = avg_mag/N
    
        Cv_sigma.append(sigma_sX(Cv_list,n_mes,2000))
        X_sigma.append(sigma_sX(X_avg,n_mes,2000))
        M_sigma.append(sigma_m((m_avg),n_mes))
        E_sigma.append(sigma_m((e_Avg),n_mes))
    
        M_list.append(np.mean(m_list)/N)
        X_1.append(X)
        Cv_1.append(CvN/temp)
        E_list.append(np.mean(e_list)/N)
        z += 1
        bar.update(z)
       
            
    bar.finish()    
#################_Visiuliastion_##################
if a == 1:
    ani_config = init(n)
    for i in T:
        t = i
        fig = plt.figure()
        im = plt.imshow(mc_k(ani_config,t,mc_step,n), animated=True)
        plt.title(t)
        
        for i in range(1000):
            updatefig()
        
        ani = animation.FuncAnimation(fig, updatefig ,interval=25, blit=True
                                      ,frames = 500,repeat = True)
        plt.show()
    ###############_data out_####################
    
if a != 1:
    
    if dynamic == 1:
        d = 'Galuber'
    else:
        d = 'Kawasaki'
    
    output = open(str(d) + '_data.txt','w')
    
    output.write('Energy\t' + 'Sigma_E\t' +'Magnitisation\t'
                 +'Sigma_M\t'+'Heat Capacity\t'+'Sigma_C\t'+'Susceptibilty\t'
                 +'Sigma_X\t\n')
    for i in range(len(T)):
        output.write(str(E_list[i]) + '\t' + str(E_sigma[i]) + '\t' + str(M_list[i]) + '\t' + str(M_sigma[i])
        + '\t' + str(Cv_1[i]) + '\t' + str(Cv_sigma[i]) + '\t' + str(X_1[i]) + '\t' + str(X_sigma[i])
        + '\n' )
        
    output.close()
    
    
    
    plt.figure()
    plt.title('Energy vs Temperature')
    plt.xlabel('Temperature')
    plt.ylabel('<E>')
    plt.errorbar(T,E_list,yerr = E_sigma,fmt='-', ecolor='orangered',color='green',
                capsize=2)#yerr = E_sigma)
    plt.savefig(str(d) + '_EvT.png')
    
    plt.figure()
    plt.title('Specific Heat Capacity vs Temperature')
    plt.xlabel('Temperature')
    plt.ylabel('<cV/N>')
    plt.errorbar(T,Cv_1,yerr=Cv_sigma,fmt='-', ecolor='orangered',color='green',
                 capsize=2)
    plt.savefig(str(d) + '_CvT.png')

    plt.figure()
    plt.title('Average magnitisation vs temperature')
    plt.xlabel('Temperature')
    plt.ylabel('<M>')
    plt.errorbar(T,M_list,yerr = M_sigma,fmt='-', ecolor='orangered',color='steelblue',
                capsize=2)#,yerr = M_sigma)
    plt.savefig(str(d) + '_MvT.png')
    
    plt.figure()
    plt.title('Magnetic Susceptibilty vs Temperature')
    plt.xlabel('Temperature')
    plt.ylabel('<X>')
    plt.errorbar(T,X_1,yerr=X_sigma,fmt='-', ecolor='orangered',color='steelblue',
                capsize=2)
    plt.savefig(str(d) + '_XvT.png')


