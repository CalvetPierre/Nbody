%matplotlib notebook
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy.linalg import norm 
from scipy.integrate import simps
import scipy.constants as pcst
plt.rcParams['figure.figsize'] = (5.0, 5.0) 

## Quelques paramètres
nCorp = 9
Msol=1.989e30 # kg masse soleil
Mmer=3.3011e23#23
Mven=4.8675e24
Mter=5.972e24 #kg masse terre
Mmar=6.4185e23
Mjup=1.8986e27 #correction
Msat=5.6846e26
Mura=8.6810e25
Mnep=102.43e24
#Mlun=7.34767309e22 # kg masse lune
Rst= 149.6e9 # m distance soleil-terre
Rtl=384400e3 # m distance terre-lune

# quelques constantes physique
G=pcst.gravitational_constant #SI
UA = 150.0e9

#Pour systeme sol Corps

m = [Msol, Mmer, Mven, Mter, Mmar, Mjup, Msat, Mura, Mnep]

# Temps de la simulation
tSimu = 365*24*3600*100

# Condition epsilon de variation de la vitesse
eps = 0.01

# Tableaus sur les masses des n-corps
tab_M = np.outer(m,m)

M = np.zeros([nCorp,nCorp])
M[:] = m

nCorp = 9

# Initialisation données de position et vitesse (x et v)
x = np.zeros([nCorp,3])
v = np.zeros([nCorp,3])

# Conditions initiales des n-Corps
#Corps,xyz
#Mercure
x[1] = np.array([-0.3936075875560, -0.0581672802186, 0.0097241052554])*UA
v[1] = np.array([-0.0023402806206, -0.0237194970911 ,-0.0124283517572])*UA/(3600*24)
#Venus
x[2] = np.array([0.3044348026187 ,-0.5956444406213 ,-0.2872765035484])*UA
v[2] = np.array([0.0182336810360 ,0.0080756592308 ,0.0024800882857])*UA/(3600*24)
#Terre
x[3] = np.array([-0.7230075877916 ,-0.6449911881528 ,-0.2795973147499])*UA
v[3] = np.array([0.0117200633339 ,-0.0113775301422 ,-0.0049326952800])*UA/(3600*24)
#Mars
x[4] = np.array([0.8258070867773 ,-1.0158748100965 ,-0.4882412249277])*UA
v[4] = np.array([0.0118132421323 ,0.0087301932440 ,0.0036856169189])*UA/(3600*24)
#Jupiter
x[5] = np.array([4.8962875522201 ,-0.7369481629587 ,-0.4350601610616])*UA
v[5] = np.array([0.0012014734479 ,0.0071778766718 ,0.0030473901081])*UA/(3600*24)
#Saturn
x[6] = np.array([7.3981121994211 ,-5.9520622196150 ,-2.7770138348888])*UA
v[6] = np.array([0.0033945973249 ,0.0039073869830 ,0.0014678548265])*UA/(3600*24)
#Uranus
x[7] = np.array([14.0566856361498 ,12.7215007118186 ,5.3727882835593])*UA
v[7] = np.array([-0.0027841842702 ,0.0023950765182 ,0.0010882160661])*UA/(3600*24)
#Neptune
x[8] = np.array([29.6817076443130 ,-3.1540912504566 ,-2.0299859703627])*UA
v[8] = np.array([0.0003696297560 ,0.0029123014542 ,0.0011829391855])*UA/(3600*24)

# Modification des vitesses pour avoir un CDM immobile
v_CDM = np.zeros([3])
for i in range(3): #xyz
    for j in range(nCorp):
        v_CDM[i] += m[j]*v[j,i]
v_CDM = v_CDM/sum(m)

for i in range(3): #xyz
    for j in range(nCorp):
        v[j,i] -= v_CDM[i] 
        
# Liste des valeurs de pas de temps
tab_t = [0]
Ldt = [0]     

# Energie potentiel, cinetique et totale du système
Ep = [0]
Ec = [0]
Etot = [0]

# Valeurs initial des énergies (p=0)
Ep[0] = E_pot(x,m)
Ec[0] = E_cin(v,m)
Etot[0] = Ep[0] + Ec[0]

FT = np.zeros([3,nCorp,nCorp])

# Integration temporelle
p = 0
datax = np.zeros([nCorp,1]).tolist()
datay = np.zeros([nCorp,1]).tolist()
dataz = np.zeros([nCorp,1]).tolist()
#print(datax)

while (tab_t[-1]<tSimu):
    
    if (p%500 == 0):
        print("p: ",p,"  Time(%): ", tab_t[-1]/tSimu*100)
        
    Dist = func_dist(x,nCorp)
    
    FT = func_FT(x,Dist,nCorp)
    
    dt = min_dt(m,v,FT,eps,nCorp)

    tab_t.append(tab_t[-1] + dt)
    
    Ldt.append(dt)
    
    for i in range(3): #xyz
        # maj vitesses
        v[:,i]=v[:,i]+dt*sum(FT[i,:]*np.divide(1,M[:]))
        
        # maj positions
        x[:,i]=x[:,i]+v[:,i]*dt
    
    if (p%10 == 0):
        for i in range(nCorp):
        #datax = np.zeros([nCorp,len(tab_t)])
        #datay = np.zeros([nCorp,len(tab_t)])
        #dataz = np.zeros([nCorp,len(tab_t)])
            datax[i].append(x[i,0])
            datay[i].append(x[i,1])
            dataz[i].append(x[i,2])
            
            
        Ep = np.append(Ep,E_pot(x,m))
        Ec = np.append(Ec,E_cin(v,m))
        Etot = np.append(Etot,Ec[-1]+Ep[-1])
    
    p += 1
print('Done')
#print(datax[0])

from mpl_toolkits.mplot3d import Axes3D

BigData =np.zeros([3,nCorp,len(datax[0])],dtype=object)
BigData[0,:] = datax[:]
BigData[1,:] = datay[:]
BigData[2,:] = dataz[:]
BigData[:,:,:] = BigData[:,:,:]/UA
BigData = np.array(BigData)

fig = plt.figure(frameon=False)
ax = fig.add_subplot(projection='3d')
ax.set_facecolor("black")

for i in range(nCorp):
    ax.plot(np.array(datax[i])/UA,np.array(datay[i])/UA,np.array(dataz[i])/UA,'o',markersize=1)

ax.grid(False)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

ax.set_xlim([-15,15])
ax.set_ylim([-15,15])
ax.set_zlim([-15,15])

ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

ax.set_xticks([]) 
ax.set_yticks([]) 
ax.set_zticks([])

ax.set_xticklabels([])
ax.set_yticklabels([])
ax.set_zticklabels([])
plt.axis('off')

plt.show()

def E_pot(x,m):
    n = len(x)
    Ep = 0
    for i in range (n):
        for j in range(i+1,n):
            Ep += -G*m[i]*m[j]/norm(x[i,:]-x[j,:])
    return Ep

def E_cin(v,m):
    Ec = 0
    n = len(v)
    for i in range (n):
        Ec += 1/2*m[i]*norm(v[i,:])**2
    return Ec

def func_FT(x,Dist,nCorp):
    
    FT = np.zeros([3,nCorp,nCorp])
    
    Norm = np.sqrt(np.square(Dist[0])+np.square(Dist[1])+np.square(Dist[2]))
    Norm_c = np.power(Norm, 3)
    
    for i in range(3):
        FT[i] = G*tab_M*Dist[i]*np.divide(1,Norm_c)
    FT = np.nan_to_num(FT, nan=0.0)
    return FT

def func_dist(x,nCorp):
    temp_x = np.zeros([nCorp,nCorp])
    temp_y = np.zeros([nCorp,nCorp])
    temp_z = np.zeros([nCorp,nCorp])
    
    temp_x[:] = x[:,0]
    temp_y[:] = x[:,1]
    temp_z[:] = x[:,2]
    
    temp_xT = temp_x.T
    temp_yT = temp_y.T
    temp_zT = temp_z.T
    
    Dist_x = temp_xT - temp_x
    Dist_y = temp_yT - temp_y
    Dist_z = temp_zT - temp_z
    # calcule des dist à mettre dehors. func avec nvx parameter dc
    Dist = [Dist_x, Dist_y, Dist_z]
    return Dist

def min_dt(m,v,FT,eps,nCorp):
    temp = np.zeros(nCorp)
    res = np.zeros([nCorp,nCorp])
    for i in range(nCorp):
        temp[i] = eps*norm(v[i])*m[i]
        
    for i in range(nCorp):
        for j in range(nCorp):
            res[i,j] = norm(FT[:,i,j])
    res[:] = np.divide(1,res[:])*temp
    # dt = min(eps*Msol*norm(vs[:,p])/norm(FT),eps*Mter*norm(vt[:,p])/norm(FT))
    return np.min(res)

#print("dtmin:", min_dt(m,v,FT,eps,9))
#disttemp = func_dist(x,3)
#func_FT(x,func_dist(x,9),9)
