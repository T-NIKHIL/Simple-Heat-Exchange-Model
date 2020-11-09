# -*- coding: utf-8 -*-

# nthota2

# This program is a dynamic simulation of a 4 particle - 4 energy level heat exchange system

# Acknowledgements :
# Acknowledgements to Tushar Nichakawade for helping me review this code 

import numpy as np
import random
import matplotlib.pyplot as plt
from plot_style import *

# -------------------------------------------------------------------------------------------------

# Input Section

# -------------------------------------------------------------------------------------------------

#Input initial configuration of each system (Select such that total energy (L + R) = 14)
#particlePosArrL gives which energy level each particle is on for the Left subsystem
#particlePosArrR gives whcih energy level each particle is on for the right subsystem
#Entries are in the order 1st, 2nd, 3rd and 4th particle
particleEnergyArrL = np.array([1,2,3,4])
particleEnergyArrR = np.array([1,1,1,1])
#Number of time steps
n = 100000

# -------------------------------------------------------------------------------------------------

# Functions used

# -------------------------------------------------------------------------------------------------

# This function generates the energy ladder from the particle energy values
# Depcits the number of particles in each energy level
def generateLadder(particleEnergyArrL,particleEnergyArrR):
    ladderL = np.zeros(shape=(4,1),dtype=int)
    ladderR = np.zeros(shape=(4,1),dtype=int)
    
    for k in range(0,4):
        count = np.count_nonzero(particleEnergyArrL == (k + 1))
        # First entry (k = 0) is no of particles in energy lev 1
        # Last entry (k = 3) is no of particles in energy lev 4
        ladderL[k] = count
        count1 = np.count_nonzero(particleEnergyArrR == (k + 1))
        ladderR[k] = count1
        
    return ladderL,ladderR

# This function allows us to get the energy of each susbsytem from the energy ladder generated from generateLadder()
def energyOfSystems(ladderL,ladderR):
    energyL = 1*ladderL[0] + 2*ladderL[1] + 3*ladderL[2] + 4*ladderL[3]
    energyR = 1*ladderR[0] + 2*ladderR[1] + 3*ladderR[2] + 4*ladderR[3]
    energyDiff = energyL - energyR
    
    return energyL, energyR, energyDiff

def calcEnergyDiff(particleEnergyArrL,particleEnergyArrR,ensemble,energyDiffArr,energyLArr,energyRArr,t):
    ladderL, ladderR = generateLadder(particleEnergyArrL, particleEnergyArrR)       
    energyL, energyR, energyDiff = energyOfSystems(ladderL, ladderR)
    microstate = np.concatenate((ladderL,ladderR), axis = 1)
    ensemble = np.concatenate((ensemble,microstate), axis = 1)
    energyDiffArr[t] = energyDiff
    energyLArr[t] = energyL
    energyRArr[t] = energyR
    
    return ensemble, energyDiffArr, energyLArr, energyRArr
      
# Energy Difference Array
energyDiffArr = np.zeros(shape=(n+1,),dtype=int)
# Energy of system L
energyLArr = np.zeros(shape=(n+1,),dtype=int)
# Energy of system R
energyRArr = np.zeros(shape=(n+1,),dtype=int)

ladderL, ladderR = generateLadder(particleEnergyArrL, particleEnergyArrR)
energyL, energyR, energyDiff = energyOfSystems(ladderL, ladderR)
ensemble = np.concatenate((ladderL,ladderR), axis = 1)

energyLArr[0] = energyL
energyRArr[0] = energyR
energyDiffArr[0] = energyDiff

print(" Energy of system L : ")
print(energyL)
print("\n")
print(" Energy of system R : ")
print(energyR)
print("\n")
print(" Energy Difference b/w systems : ")
print(energyDiff)
print("\n")

print(" Microstate at t = 0 : ")

print(ensemble)

particles = np.array([1,2,3,4])
pairSelection = np.array([0,1,2])
coinToss = np.array([0,1])

# -------------------------------------------------------------------------------------------------

# Loop Section

# -------------------------------------------------------------------------------------------------
    
for i in range(1,n+1):
    
    # --------------------------------------------------------------
    
    # STEP 1 : Select pair of particles randomly from same subsystem 
    #          or opposite subsystems and obtain their energy levels
    
    # --------------------------------------------------------------
    
    # The two particles in the pair are called particle A and particle B
    # Their energies are given by energy A and energy B
    
    #This variable is used to determine if pair of particles are from the same system or different systems
    #If toss1 == 0 (Particles from system L) 
    #If toss1 == 1 (Particles from system R)
    #If toss1 == 2 (Particles from different systems)
    
    toss1 = random.choice(pairSelection)
    
    # print("Toss 1:")
    # print(toss1)
    
    if (toss1 == 0):
        # Choosing the position/energy arrays to use for each particle
        particleEnergyArrA = particleEnergyArrL
        particleEnergyArrB = particleEnergyArrL  
        
        while True:
            particleA = random.choice(particles)
            particleB = random.choice(particles)
            if (particleA != particleB):
                break
        
    elif (toss1 == 1):
        # Choosing the position/energy arrays to use for each particle
        particleEnergyArrA = particleEnergyArrR
        particleEnergyArrB = particleEnergyArrR  
        
        while True:
            particleA = random.choice(particles)
            particleB = random.choice(particles)
            if (particleA != particleB):
                break
        
    else :
        # Choosing the position/energy arrays to use for each particle
        particleEnergyArrA = particleEnergyArrL
        particleEnergyArrB = particleEnergyArrR 
        
        #Random selection of first particle of the pair
        particleA = random.choice(particles)
        #Random selection of second particle of the pair
        particleB = random.choice(particles)
        
    
    particleEnergyA = particleEnergyArrA[particleA - 1]
    particleEnergyB = particleEnergyArrB[particleB - 1]
            
    # --------------------------------------------------------------
    
    # STEP 2 : Increase / decrease energies of particles in the pair
    
    # --------------------------------------------------------------
        
    # Case 1 : When both particles are in the highest or lowest energy levels
    if (particleEnergyA == 1) and (particleEnergyB == 1):
        
        ensemble, energyDiffArr, energyLArr, energyRArr = calcEnergyDiff(particleEnergyArrA,particleEnergyArrB,ensemble,energyDiffArr,energyLArr,energyRArr,i) 
        if (i % 10000) == 0:
                print(".")
        continue
    
    elif (particleEnergyA == 4) and (particleEnergyB == 4):
        
        ensemble, energyDiffArr, energyLArr, energyRArr = calcEnergyDiff(particleEnergyArrA,particleEnergyArrB,ensemble,energyDiffArr,energyLArr,energyRArr,i)
        if (i % 10000) == 0:
                print(".")
        continue      
    
    elif (particleEnergyA == 4):
        
        particleEnergyArrA[particleA - 1] = particleEnergyA - 1
        particleEnergyArrB[particleB - 1] = particleEnergyB + 1
        
        ensemble, energyDiffArr, energyLArr, energyRArr = calcEnergyDiff(particleEnergyArrA,particleEnergyArrB,ensemble,energyDiffArr,energyLArr,energyRArr,i)
        
        if (toss1 == 0):
            particleEnergyArrL = particleEnergyArrA
        elif (toss1 == 1):
            particleEnergyArrR = particleEnergyArrA
        else:
            particleEnergyArrL = particleEnergyArrA
            particleEnergyArrR = particleEnergyArrB
        
        if (i % 10000) == 0:
            print(".")
        continue
    
    elif (particleEnergyA == 1):
        
        particleEnergyArrA[particleA - 1] = particleEnergyA + 1
        particleEnergyArrB[particleB - 1] = particleEnergyB - 1
        
        ensemble, energyDiffArr, energyLArr, energyRArr = calcEnergyDiff(particleEnergyArrA,particleEnergyArrB,ensemble,energyDiffArr,energyLArr,energyRArr,i)
        
        if (toss1 == 0):
            particleEnergyArrL = particleEnergyArrA
        elif (toss1 == 1):
            particleEnergyArrR = particleEnergyArrA
        else:
            particleEnergyArrL = particleEnergyArrA
            particleEnergyArrR = particleEnergyArrB
        
        if (i % 10000) == 0:
            print(".")
        continue
    
    elif (particleEnergyB == 4):
        
        particleEnergyArrA[particleA - 1] = particleEnergyA + 1
        particleEnergyArrB[particleB - 1] = particleEnergyB - 1
        
        ensemble, energyDiffArr, energyLArr, energyRArr = calcEnergyDiff(particleEnergyArrA,particleEnergyArrB,ensemble,energyDiffArr,energyLArr,energyRArr,i)
        
        if (toss1 == 0):
            particleEnergyArrL = particleEnergyArrA
        elif (toss1 == 1):
            particleEnergyArrR = particleEnergyArrA
        else:
            particleEnergyArrL = particleEnergyArrA
            particleEnergyArrR = particleEnergyArrB
        
        if (i % 10000) == 0:
            print(".")
        continue
    
    elif (particleEnergyB == 1):
        
        particleEnergyArrA[particleA - 1] = particleEnergyA - 1
        particleEnergyArrB[particleB - 1] = particleEnergyB + 1
        
        ensemble, energyDiffArr, energyLArr, energyRArr = calcEnergyDiff(particleEnergyArrA,particleEnergyArrB,ensemble,energyDiffArr,energyLArr,energyRArr,i)
        
        if (toss1 == 0):
            particleEnergyArrL = particleEnergyArrA
        elif (toss1 == 1):
            particleEnergyArrR = particleEnergyArrA
        else:
            particleEnergyArrL = particleEnergyArrA
            particleEnergyArrR = particleEnergyArrB
        
        if (i % 10000) == 0:
            print(".")
        continue
    
    else :
        
    # Case 3 : Both particles in intermediate energy levels
        
    # Lets do a toss to figure out which particle's energy has to be increased
    # If toss2 == 0 (increment energy of particle A) ; if toss2 == 1 (increment energy of particle B)
    
        toss2 = random.choice(coinToss)
        
        if toss2 == 0:
            particleEnergyArrA[particleA - 1] = particleEnergyA + 1
            particleEnergyArrB[particleB - 1] = particleEnergyB - 1
        
            ensemble, energyDiffArr, energyLArr, energyRArr = calcEnergyDiff(particleEnergyArrA,particleEnergyArrB,ensemble,energyDiffArr,energyLArr,energyRArr,i)
            
            if (toss1 == 0):
                particleEnergyArrL = particleEnergyArrA
            elif (toss1 == 1):
                particleEnergyArrR = particleEnergyArrA
            else:
                particleEnergyArrL = particleEnergyArrA
                particleEnergyArrR = particleEnergyArrB
            
            if (i % 10000) == 0:
                print(".")
            continue
        
        else :
            particleEnergyArrA[particleA - 1] = particleEnergyA - 1
            particleEnergyArrB[particleB - 1] = particleEnergyB + 1
        
            ensemble, energyDiffArr, energyLArr, energyRArr = calcEnergyDiff(particleEnergyArrA,particleEnergyArrB,ensemble,energyDiffArr,energyLArr,energyRArr,i)
            
            if (toss1 == 0):
                particleEnergyArrL = particleEnergyArrA
            elif (toss1 == 1):
                particleEnergyArrR = particleEnergyArrA
            else:
                particleEnergyArrL = particleEnergyArrA
                particleEnergyArrR = particleEnergyArrB
            
            if (i % 10000) == 0:
                print(".")
            continue
        
# -------------------------------------------------------------------------------------------------

# Plotting Section

# -------------------------------------------------------------------------------------------------   

print(" Plotting ... ")

# FIGURE 1 : Time trajectories of EL - ER
    
plot_style1(np.arange(0,101,1),energyDiffArr[0:101], "Time trajectories for 100 time steps ", "Time Step", "EL - ER")
plot_style1(np.arange(0,1001,1),energyDiffArr[0:1001], "Time trajectories for 1000 time steps ", "Time Step", "EL - ER")     
plot_style1(np.arange(0,10001,1),energyDiffArr[0:10001], "Time trajectories for 10000 time steps ", "Time Step", "EL - ER") 

# FIGURE 2 : Plotting Auto-Correlation function for EL

y_avg = np.mean(energyLArr)
tau = np.arange(0,100)
CyArr = np.zeros(shape=(100,),dtype=float)
den = 0

for j in tau:
    
    # Reset steps
    l = 0
    num = 0
    
    while True:
        
        num = num + (energyLArr[l] - y_avg)*(energyLArr[l + j] - y_avg)
        if j == 0 :
            den = den + (energyLArr[l] - y_avg)*(energyLArr[l] - y_avg)          
        l = l + 1         
        if (l + j > n) :
            break
    
    Cy = num/den
    CyArr[j] = Cy    

plot_style1(tau,CyArr, "Autocorrelation function for EL", "Tau", "Autocorrelation of EL")

# FIGURE 3 : Plotting Histogram of Energy values of left system and right system

fig1, ax1 = plt.subplots()
fig2, ax2 = plt.subplots()
fig3, ax3 = plt.subplots()


hist1 = ax1.hist(energyLArr,[4,5,6,7,8,9,10,11],histtype='bar',orientation='vertical')
ax1.set_xticks([4,5,6,7,8,9,10,11])
ax1.set_xticklabels([4,5,6,7,8,9,10,11])
ax1.set_xlabel('Energy of system L')
ax1.set_ylabel('Counts')
ax1.set_title('Distribution of Energy Values of system L')

mu_L = np.mean(energyLArr)
sigma_L = np.std(energyLArr)

textstr1 = '\n'.join((r'$\mu=%.2f$' % (mu_L, ),r'$\sigma=%.2f$' % (sigma_L, )))

ax1.text(0.05, 0.95, textstr1, transform=ax1.transAxes, fontsize=14,verticalalignment='top')

for i in range(7):
	ax1.text(hist1[1][i] + 0.1,hist1[0][i] + 0.1,str(hist1[0][i]))
    
# ---------------------------------------------------------------------------------------

hist2 = ax2.hist(energyRArr,[4,5,6,7,8,9,10,11],histtype='bar',orientation='vertical')
ax2.set_xticks([4,5,6,7,8,9,10,11])
ax2.set_xticklabels([4,5,6,7,8,9,10,11])
ax2.set_xlabel('Energy of system R')
ax2.set_ylabel('Counts')
ax2.set_title('Distribution of Energy Values of system R')

mu_R = np.mean(energyRArr)
sigma_R = np.std(energyRArr)

textstr2 = '\n'.join((r'$\mu=%.2f$' % (mu_R, ),r'$\sigma=%.2f$' % (sigma_R, )))

ax2.text(0.05, 0.95, textstr2, transform=ax2.transAxes, fontsize=14,verticalalignment='top')

for i in range(7):
	ax2.text(hist2[1][i] + 0.1,hist2[0][i] + 0.1,str(hist2[0][i]))
    
# ---------------------------------------------------------------------------------------

hist3 = ax3.hist(energyDiffArr,[-6,-4,-2,0,2,4,6],histtype='bar',orientation='vertical')
ax3.set_xticks([-6,-4,-2,0,2,4,6])
ax3.set_xticklabels([-6,-4,-2,0,2,4,6])
ax3.set_xlabel('EL - ER')
ax3.set_ylabel('Counts')
ax3.set_title('Distribution of EL - ER')

mu_diff = np.mean(energyDiffArr)
sigma_diff = np.std(energyDiffArr)

textstr3 = '\n'.join((r'$\mu=%.2f$' % (mu_diff, ),r'$\sigma=%.2f$' % (sigma_diff, )))

ax3.text(0.05, 0.95, textstr3, transform=ax3.transAxes, fontsize=14,verticalalignment='top')

for i in range(7):
	ax3.text(hist3[1][i] + 0.1,hist3[0][i] + 0.1,str(hist3[0][i]))

plt.show()






        
        

    
        
        
        

    
