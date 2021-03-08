import os
import sys
sys.path.append('../')
import gridemic
import numpy as np
from scipy.optimize import root_scalar

R0 = float(sys.argv[1])
print(R0)

# Common settings
Seed     = 5772   # A seed for the RNG. None lets numpy.random select a seed randomly
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
NTime       = 10000  # Maximum duration of the simulation
num_tests   = 1000   # Number of tests available per time step
test_begin  = 0      # Day when Testing and Quarantine starts

def fR0(tau, R0):

    return 4*tau+4*(1-(1-tau)**4)-R0*1.33

if R0 == 0:

    tau =0 

else: 
    
    tau = root_scalar(fR0, args=(R0), bracket = [1e-6, 1]).root


SSEM = gridemic.Model(
    seed_random  = Seed, 
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


Population, _ = SSEM.simulate(NTime = NTime, num_initial_infectious = NIi)

np.savetxt('data/Pop_R0={:04.2f}_NTests=1000.dat'.format(R0),Population)