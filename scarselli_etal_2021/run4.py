import os
import sys
sys.path.append('../')
import gridemic
import numpy as np
from scipy.optimize import root_scalar

eps = float(sys.argv[1])

# Parameters
Seed     = 5772   # A seed for the RNG. None lets numpy.random select a seed randomly
N        = 3162   # Grid size. N x N = P
NTime    = 600    # Maximum duration of the simulation
dt       = 1      # Time step
NIi      = 100    # Number of initial weak-symptom infectious
kEI      = 3      # E---> I shape parameter for the Gamma distribution
thetaEI  = 1      # E---> I scale parameter for the Gamma distribution
kIR      = 4      # I---> R shape parameter for the Gamma distribution
thetaIR  = 1      # I---> R scale parameter for the Gamma distribution
prob_symptom = 0.50   # Probability of developing strong symptoms
num_tests   = 1000   # Number of tests available per time step
prob_trace   = 1.00   # Probability of succeeding in tracing a neighbor
prob_detect   = 1.00   # Probability of succeeding in recognizing a strong-symptom case
test_lag  = 1      # Time to wait before knowing the test result
test_begin    = 20     # Day when Testing and Quarantine starts
verbose  = 0       # Verbose output if 1

R0  = 2.0

def fR0(tau, R0):

    return 4*tau+4*(1-(1-tau)**4)-R0*1.33
    
tau = root_scalar(fR0, args=(R0), bracket = [1e-6, 1]).root

Rt = np.zeros(NTime)
    
# Duration of lockdown
NLD = 30

SSEM = gridemic.Model(
    seed_random  = None, 
    N            = N, 
    kEI          = kEI, 
    thetaEI      = thetaEI,
    kIR          = kIR,
    thetaIR      = thetaIR, 
    prob_symptom = prob_symptom,
    tauW         = tau, 
    etaW         = tau,
    tauS         = tau,
    etaS         = tau,
    num_tests    = num_tests,
    prob_trace   = prob_trace,
    prob_detect  = prob_detect,
    test_lag     = test_lag,
    test_begin   = test_begin        
)

SSEM.add_infectious(num_initial_infectious=NIi)

Population = np.zeros((NTime + 1, 6))    

while SSEM.time < NTime:
    
    if SSEM.time % 10 == 0:
        print(SSEM.time, end=', ')
    
    Population[SSEM.time, 0] = np.sum(SSEM.disease_state==0) # S
    Population[SSEM.time, 1] = np.sum(SSEM.disease_state==1) # E
    Population[SSEM.time, 2] = np.sum(SSEM.disease_state==2) # I_w
    Population[SSEM.time, 3] = np.sum(SSEM.disease_state==3) # I_s
    Population[SSEM.time, 4] = np.sum(SSEM.disease_state==4) # I_R
    Population[SSEM.time, 5] = (np.sum(SSEM.testing_state==3) 
                        + np.sum(SSEM.testing_state==4)) # cases   
    
    if SSEM.time >= test_begin and SSEM.time < test_begin + NLD:
        
        SSEM.tauW = tau * eps
        SSEM.etaW = tau * eps
        SSEM.tauS = tau * eps
        SSEM.etaS = tau * eps
    
    else:

        SSEM.tauW = tau
        SSEM.etaW = tau
        SSEM.tauS = tau
        SSEM.etaS = tau

        
    # Stop if no susceptible left
    if Population[SSEM.time, 0] == 0:

        if verbose:

            print("Everyone is infected")

        break

    # Stop if no infectious

    if (Population[SSEM.time, 1] == 0 
    and Population[SSEM.time, 2] == 0  
    and Population[SSEM.time, 3] == 0):

        if verbose:

            print("No active infection")

        break            
        
    SSEM.evolve()

np.savetxt('data/Pop_eps={:04.2f}_NLD=30.dat'.format(eps),Population)