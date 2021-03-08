import os
import sys
sys.path.append('../')
import gridemic
import numpy as np
from scipy.optimize import root_scalar

k = int(sys.argv[1])
print(k)

# Common settings
N        = 3162   # Grid size. N x N = P
NIi      = 100    # Number of initial weak-symptom infectious
kEI      = 3      # E---> I shape parameter for the Gamma distribution
thetaEI  = 1      # E---> I scale parameter for the Gamma distribution
kIR      = 4      # I---> R shape parameter for the Gamma distribution
thetaIR  = 1      # I---> R scale parameter for the Gamma distribution
prob_symptom = 0.50   # Probability of developing strong symptoms
test_lag  = 1      # Time to wait before knowing the test result (if testing is on)
verbose  = 0      # Verbose output if 1

# Parameters
prob_trace  = 1.00   # Probability of succeeding in tracing a neighbor
prob_detect = 1.00   # Probability of succeeding in recognizing a strong-symptom case
NTime       = 200  # Maximum duration of the simulation
num_tests   = 1000     # Number of tests available per time step
test_begin  = 0  # Day when Testing and Quarantine starts

R0  = 2.7

def fR0(tau, R0):

    return 4*tau+4*(1-(1-tau)**4)-R0*1.33
    
tau = root_scalar(fR0, args=(R0), bracket = [1e-6, 1]).root

Rt = np.zeros(NTime)

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

_, _ = SSEM.simulate(NTime = NTime, num_initial_infectious = NIi)

for it in range(NTime):
    infected_today = np.argwhere(SSEM.infection_day == it)
    NInf = infected_today.shape[0]
    new_infections = np.zeros(NInf)
    for j in range(NInf):
        new_infections[j] = SSEM.new_infections[infected_today[j,0], infected_today[j,1]]
    Rt[it] = np.mean(new_infections)
np.savetxt('data/Rt_R0=2.7_k={}.dat'.format(k),Rt)