#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 22 13:10:30 2018

@author: pichugina
"""
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 09:49:18 2018
Control kinetic model without cellulose
@author: pichugina
"""
import sys
from math import exp, fsum, ceil
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import cg
from datetime import datetime


###############################################################################
# Model parameters 
###############################################################################
# S A types
WA=0 #3.7e-4       # [1/s]  A-attached type grownth rate
WS=0 #3.7e-4       # [1/s]  S-swimming type grownth rate
Xlength=350        # [mkm] total length of the layer in mkm
P=0.5              # [] adsorbtion probability
D0=0.1             # [mkm2/s] diffusion constant
###########
# cellulose
###########
Dcl=10000           # [mkm2/s] diffusion constant
ClProduction=1000  # [1/s]
k=1000000

#Simulation parameters
Nlayers=10
dx=Xlength/Nlayers
X=[(i+1)*dx for i in range(Nlayers)]

dt=0.01
NtimeSteps=20
LambdaCl=Dcl*dt/(dx*dx)

###############################################################################
# Initial conditions
###############################################################################
Nfreq=1    # Saving data to the file # how frequent to save               
NTsave=NtimeSteps/Nfreq
T=[i*Nfreq for i in range(NTsave+1)]
Total=np.zeros((NTsave,1))      
       

# S-initial condition as gauss
SProfile=np.zeros((Nlayers,NTsave))
Stotal=np.zeros((NTsave,1))
mu=Xlength/2.0
sigma=Xlength*0.1
S0=100.0
Sprev=[S0*exp(-(X[i]-mu)**2/sigma/sigma) for i in range(Nlayers)]
SprevSum=fsum(Sprev)
Sprev=[Sprev[i]/SprevSum for i in range(Nlayers)]
TotalStep0=np.sum(Sprev)*dx
#plt.plot(X,Sprev)
#plt.show()


# A-initial condition
A=np.zeros((NTsave,3))  #  3colums array A-sep A-grown A-transition 
Aprev=0
# Initial condition
SProfile[0:Nlayers,0]=Sprev
Total[0]=0

# Cl initial condition
ClProfile=np.zeros((Nlayers,NTsave))
Clprev=np.zeros((Nlayers,1))
Clnext=np.zeros((Nlayers,1))
DProfile=np.zeros((Nlayers,NTsave))
Dnext=D0*np.ones((Nlayers,1)) 

###############################################################################
# Backward Euler scheme
###############################################################################

################
## cellulose ###
################
# Backward Euler scheme 
# Coefficient matrix
# Coefficient matrix
Clmatrix=np.zeros((Nlayers,Nlayers))
Lambda=np.zeros((Nlayers,1))
Smatrix=np.zeros((Nlayers,Nlayers))
counter=1;


## Cellulose matrix
##### Update cellulose
for i in range(1,Nlayers-1):
    Clmatrix[i,i-1]=-LambdaCl
    Clmatrix[i,i]=(1+2*LambdaCl)
    Clmatrix[i,i+1]=-LambdaCl
    
# Boundary conditions
Clmatrix[0,0]=1+LambdaCl  # Reflective boundary
Clmatrix[0,1]=-LambdaCl
Clmatrix[Nlayers-1,Nlayers-2]=-LambdaCl # Reflective boundary
Clmatrix[Nlayers-1,Nlayers-1]=1+LambdaCl 
    

for i in range(1,NtimeSteps):
    Clprev[0]=Clprev[0]+ClProduction*Aprev*dt
    Clnext=cg(Clmatrix,Clprev,tol=1e-12)[0]
    # from Cl concentration to the D
    Dnext=D0*1/(1+k*Clnext)
    
    ##### Update S-type and A-type
    Lambda=Dnext*dt/(dx*dx)
    print(Lambda)
    
    # Coefficient matrix
    for i in range(1,Nlayers-1):
        Smatrix[i,i-1]=-Lambda[i]
        Smatrix[i,i]=(1+Lambda[i+1]+Lambda[i]-WS*dt)
        Smatrix[i,i+1]=-Lambda[i+1]
        
     # Boundary conditions
    Smatrix[0,0]=1+Lambda[1]+Lambda[0]-Lambda[0]*(1-P)-WS*dt # Adsorbing boundary 
    Smatrix[0,1]=-Lambda[1]
    Smatrix[Nlayers-1,Nlayers-2]=-Lambda[Nlayers-1] # Reflective boundary
    Smatrix[Nlayers-1,Nlayers-1]=1+Lambda[Nlayers-1]  
    
    # Solve matrix equations Smatrix
    Snext=cg(Smatrix,Sprev,tol=1e-12)[0]
    Anext=(P*dx*Lambda[0]*Snext[0]+Aprev)/(1-WA*dt) #
        
    # save profile every Nfreq steps
    if (i % Nfreq)==0:
        print(i)
        SProfile[0:Nlayers,counter]=Snext
        ClProfile[0:Nlayers,counter]=Clnext
        DProfile[0:Nlayers,counter]=Dnext
        Stotal[counter]=fsum(Snext)*dx
        A[counter,0]=Anext
        A[counter,1]=WA*Anext*dt
        A[counter,2]=P*dx*Lambda[0]*Snext[0]
        Total[counter]=Anext+fsum(Snext)*dx-TotalStep0
        counter=counter+1
        
    # update
    Aprev=Anext
    Sprev=Snext
    Clprev=Clnext
###############################################################################
# Save to file    
###############################################################################
plt.plot(SProfile)
plt.title('SProfile')
plt.show()

plt.plot(np.sum(ClProfile,1))
plt.title('ClProfile)')
plt.show()    
    
plt.plot(DProfile)
plt.title('DProfile') 
plt.show()

plt.plot(A[:,0])
plt.title('Atype')
plt.show()
   
# files with profile
root_folder='/Users/pichugina/Work/KineticModel_python/Test/'
datastamp=datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
filename_Sprofile=root_folder+datastamp+'__Stype_profile.txt'
filename_A=root_folder+datastamp+'__Atype_profile.txt';
filename_Total=root_folder+datastamp+'__Total.txt';
filename_STotal=root_folder+datastamp+'__STotal.txt';
filename_STotal=root_folder+datastamp+'__STotal.txt';
filename_CLprofile=root_folder+datastamp+'__Clprofile.txt';

np.savetxt(filename_Sprofile,SProfile,delimiter='\t')
np.savetxt(filename_A,A,delimiter='\t')
np.savetxt(filename_Total,Total,delimiter='\t')
np.savetxt(filename_STotal,Stotal,delimiter='\t')
np.savetxt(filename_CLprofile,ClProfile,delimiter='\t')

#file with parameters
filename_parameters=root_folder+datastamp+'__ParametersFile.txt';
Text=[]
Text.append("WA\t%4.10f\t1/sec\t\n" %WA)
Text.append("WS\t%4.10f\t1/sec\t \n" %WS)
Text.append("Xlength\t%4.4f\tmkm\t\n" %Xlength)
Text.append("D\t%4.4f\tmkm^2/s\t\n" %D0)
Text.append("P\t%4.4f\t[]\t\n" %P)
Text.append("dt\t%4.4f\ts\t\n" %dt)
Text.append("Nlayers\t%4.4f\t[]\t\n" %Nlayers)
Text.append("Nfreq\t%4.4f\t[]\t\n" % Nfreq)
Text.append("NtimeSteps\t%4.4f\t[]\t\n" %NtimeSteps)

f = open(filename_parameters, 'a')
for line in Text:
    #print(line)
    f.write(line)
f.close()
###############################################################################








