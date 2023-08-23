import numpy as np 
from numpy.random import rand, randint, gamma, choice, shuffle, seed
import seaborn as sns
from matplotlib import pyplot as plt 
from matplotlib import cm
import matplotlib as mpl

class Model():
    """
    Agent-based epidemic model on a square grid with a testing, contact 
    tracing, and quarantine response coupled to the disease dynamics. 
    
    Agents are identified by their positions on the grid and are described by 
    two state variables: (1) disease state, (2) testing state. 

    Disease states
    --------------
    0: Susceptible, 1: Exposed, 2: Weak-symptom infectious, 
    3: Strong-symptom infectious, 4: Removed

    Testing states
    --------------
    0: Not tested, 1: Suspect: will be tested if tests available
    2: Tested, 3: Positive in contact tracing, 
    4: positive, contacts already traced
    """


    def __init__(self, seed_random = None, N = 500, kEI = 3, thetaEI = 1, 
                 kIR = 4, thetaIR = 1, prob_symptom = 0.5, tauW = 0.5, etaW = 0.5, 
                 tauS = 0.5, etaS = 0.5, num_tests = int(250e3), prob_trace = 0.5, 
                 prob_detect = 0.5, test_lag = 1, test_begin = 0):
        """
        Parameters:
        -----------
        seed_random : int
            seed_random for the RNG. None lets numpy.random select randomly
        N : int
            Grid size. N x N = P
        kEI : float
            E---> I shape parameter for the Gamma distribution
        thetaEI : float
            E---> I scale parameter for the Gamma distribution
        kIR : float
            I---> R shape parameter for the Gamma distribution
        thetaIR : float
            I---> R scale parameter for the Gamma distribution
        prob_symptom : float
            probability [0, 1] of developing strong symptoms
        tauW : float
            probability [0, 1] of a weak-symptom infectious transmitting the 
            disease to a neighbor in a dt 
        etaW : float
            Probability [0, 1] of a weak-symptom infectious transmitting the 
            disease to a random person
        tauS : float
            Probability [0, 1] of a strong-symptom infectious transmitting the 
            disease to a neighbor
        etaS : float 
            Probability [0, 1] of a strong-symptom infectious transmitting the 
            disease to a random person
        num_tests : int
            Number of tests available per time step
        prob_trace : float
            Probability [0, 1] of tracing a neighbor
        prob_detect : float
            Probability [0, 1] of detecting a strong-symptom case
        test_lag : float
            Time to wait before knowing the test result
        test_begin : int
            Time step at which test and quarantine starts
        """
        self.seed_random  = seed_random
        self.N            = N
        self.kEI          = kEI
        self.thetaEI      = thetaEI
        self.kIR          = kIR
        self.thetaIR      = thetaIR
        self.prob_symptom = prob_symptom
        self.tauW         = tauW
        self.etaW         = etaW
        self.tauS         = tauS
        self.etaS         = etaS
        self.num_tests    = num_tests
        self.prob_trace   = prob_trace
        self.prob_detect  = prob_detect
        self.test_lag     = test_lag
        self.test_begin   = test_begin

        # Main arrays describing the population
        self.disease_state = np.zeros((N, N)) 
        self.testing_state = np.zeros((N, N)) 
        
        # Auxiliary arrays for dynamics
        self.test_results   = np.zeros((N, N))       
        self.new_infections = np.zeros((N, N)) # counter
        self.infection_day  = np.zeros((N, N)) * np.nan  
        self.disease_clock  = np.zeros((N, N)) # latent and infectious clocks
        self.test_clock     = np.zeros((N, N)) # clock testing
        self.duration       = np.zeros((N, N)) # durations in E and/or I states
        self.tests_today    = 0

        self.time = 0
        seed(self.seed_random)    


    def add_infectious(self, state=2, num_initial_infectious = 3):
        """
        Add infectious individuals to the population located at 
        random grid points. 
        
        Parameters
        ----------        
        state : int 
            (2: weak-symptom, 3: strong symptom)
        num_initial_infectious : int
            number of people in state I_W (2) at time 0
        """

        infection = 0
        while infection < num_initial_infectious: 

            susceptible = 1 
            
            while not(susceptible == 0):
        
                # Select a random person on the grid
                i_random = randint(self.N)
                j_random = randint(self.N)
                
                susceptible = self.disease_state[i_random, j_random] 

            self.disease_state[i_random, j_random] = state 
            self.infection_day[i_random, j_random]         = self.time
    
            # Timing of the infection
            self.disease_clock[i_random, j_random] = self.time
            self.duration[i_random, j_random] = (
                gamma(self.kIR, self.thetaIR) - 0.5
                ) # Factor 0.5 is necessary to offset the discrete 
                  # time step
            
            infection += 1


    def visualize_population(self, ax=None, show_legend=True):
        """
        visualize the current state of the population in a graph
        """
        # Colorblind-accessible palette
        colpal = sns.color_palette("colorblind", 4)
        colmap = cm.get_cmap('Set2', 5)
        colmap.colors[0] = np.array([1.0, 1.0, 1.0, 1.0]) # susceptible: white
        
        for k in range(1, 5):
            colmap.colors[k] = np.array([colpal[4-k][0],
                                         colpal[4-k][1],
                                         colpal[4-k][2], 1.0]) 

        if ax == None:
            fig = plt.figure(figsize=(6,6))
            ax = fig.gca()            

        im = ax.imshow(self.disease_state,
                       cmap=colmap,
                       vmin=-0.5,
                       vmax=4.5,
                       aspect='equal')
        
        if show_legend:
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_yticks([0, 1, 2, 3, 4])
            cbar.ax.set_yticklabels(['$S$', '$E$', '$I_W$', '$I_S$', '$R$'])


        ax.contour(self.testing_state, [1, 2, 3, 4])

        ax.tick_params(axis='both',
                       which='both',
                       bottom=False,
                       top=False,
                       labelbottom=False,
                       left=False,
                       right=False,
                       labelleft=False)

        return ax


    def evolve(self):
        """
        Take one time step
        """

        def get_neighbors(individual):
            
            neighbors = [[(individual[0] - 1)%self.N, individual[1]],
                         [(individual[0] + 1)%self.N, individual[1]],
                         [individual[0], (individual[1] - 1)%self.N],
                         [individual[0], (individual[1] + 1)%self.N]
                         ]
            return neighbors


        def is_ssusceptible(individual):
            """
            return True if the individual is susceptible and not under 
            isolation due to being a contact. Misspelling of the function name
            is intentional to distinguish from the epidemiological state.
            """

            ssusceptible = (
                self.disease_state[individual[0], individual[1]] == 0
                and self.testing_state[individual[0], individual[1]] == 0)
            
            return ssusceptible


        def interact_neighbors(individual, prob_transmission):
            
            neighbors = get_neighbors(individual)

            for neighbor in neighbors:
                
                if (is_ssusceptible(neighbor) and rand() < prob_transmission):

                    # transmission successful, neighbor is exposed
                    self.disease_state[neighbor[0], neighbor[1]] = 1 
                    self.disease_clock[neighbor[0], neighbor[1]] = 0 
                    self.duration[neighbor[0], neighbor[1]] = (
                        gamma(self.kEI, self.thetaEI) - 0.5 ) # Latent dur.
                                                
                    self.new_infections[individual[0], individual[1]] += 1       
                    self.infection_day[neighbor[0], neighbor[1]] = self.time


        def interact_random(individual, prob_transmission):

            # Infect a random person on the grid
            random = [randint(self.N), randint(self.N)]
            
            if (is_ssusceptible(random) and rand() < prob_transmission): 

                    # transmission successful, random is exposed
                    
                self.disease_state[random[0], random[1]] = 1 
                self.disease_clock[random[0], random[1]] = 0 # reset clock
                self.duration[random[0], random[1]] = (
                    gamma(self.kEI, self.thetaEI) - 0.5) # Latent duration
                                            
                self.new_infections[individual[0], individual[1]] += 1       
                self.infection_day[random[0], random[1]] = self.time

        def advance_disease(individual):

            state = self.disease_state[individual[0], individual[1]]
            self.disease_clock[individual[0], individual[1]] += 1

            # Remove if the clock exceeds the duration 
            if (self.disease_clock[individual[0], individual[1]] 
              > self.duration[individual[0], individual[1]]):

                if state == 2 or state == 3:
                    # I -> R transition
                    self.disease_state[individual[0], individual[1]] = 4
                
                elif state == 1:
                    # E -> I transition
                    if (rand() < self.prob_symptom):
                        # Strong-symptom infectious
                        self.disease_state[individual[0], individual[1]] = 3
                    else:
                        # Weak-symptom infectious
                        self.disease_state[individual[0], individual[1]] = 2

                    # Reset the disease clock & assign infectious time
                    self.disease_clock[individual[0], individual[1]] = 0 
                    self.duration[individual[0], individual[1]] = (
                        gamma(self.kIR, self.thetaIR) - 0.5)

        # ------ Contact tracing and recognition of strong-symptom cases ------
        
        # Activate Testing and Quarantine after test_begin
        if self.time >= self.test_begin:
            
            # List new positive cases
            positives = np.argwhere(self.testing_state == 3)
            
            for individual in positives:              
                
                neighbors = get_neighbors(individual)
                
                for neighbor in neighbors:
                    if self.testing_state[neighbor[0], neighbor[1]] == 0:
                        # the neighbor is not already tested
                        if rand() < self.prob_trace:
                            # Contact tracing success
                            # The neighbor is now a suspect
                            self.testing_state[neighbor[0], neighbor[1]] = 1                        

                # Set case status to "contacts traced"
                self.testing_state[individual[0], individual[1]]= 4

            # Strong-symptom cases that are have not been tested before
            symptomatics     = np.argwhere(
                            np.logical_and(self.disease_state == 3, 
                                           self.testing_state == 0)
                            )

            for individual in symptomatics :

                if rand() < self.prob_detect:
                    # symptom recognition success
                    # identify the individual as a suspect
                    self.testing_state[individual[0], individual[1]] = 1 
        

        # ------ Weak-symptom infectious ------

        # List weak-symptom infectious that are not suspects or tested
        asymptomatics = np.argwhere(np.logical_and(self.disease_state == 2, 
                                                   self.testing_state == 0))
    
        for individual in asymptomatics:
            
            # Interactions that might result in new infections
            interact_neighbors(individual, self.tauW)
            interact_random(individual, self.etaW)

            advance_disease(individual) 
                
        # ------ Strong-symptom infectious ------  

        # List strong-symptom infectious that are not suspects or tested
        symptomatics  = np.argwhere(np.logical_and(self.disease_state == 3, 
                                                   self.testing_state == 0))
            
        for individual in symptomatics :   

            interact_neighbors(individual, self.tauS)
            interact_random(individual, self.etaS)

            advance_disease(individual)
            
        # ------ Test suspects ------
        
        # List of suspects
        suspects = np.argwhere(self.testing_state == 1)
        
        # Shuffle the order assuming we do not give priority testing to anyone
        shuffle(suspects)

        self.tests_today = 0 # test counter
        for suspect in suspects:
            
            # Are there test available?
            if self.tests_today < self.num_tests:
                
                # Store the suspect status for later evaluation
                self.test_results[suspect[0], suspect[1]] = \
                    self.disease_state[suspect[0],suspect[1]].copy()
                
                # Mark the suspect as tested
                self.testing_state[suspect[0], suspect[1]] = 2
                
                # Start the timer
                self.test_clock[suspect[0], suspect[1]] = 0
                
                # One test kit less available
                self.tests_today += 1
                
            else:
                # We are done for the day
                break
 
        # ------ Tested people: check result ------     
        
        # List of people that have been tested
        tested = np.argwhere(self.testing_state==2)
        
        for individual in tested:
            
            # Select the test result of the current person
            result = self.test_results[individual[0], individual[1]].copy()
                    
            # Is it time to check the result?
            if self.test_clock[individual[0], individual[1]] >= self.test_lag:
                
                # Did the person catch the disease?
                if (result == 1 or result == 2 or result == 3 or result == 4):
                
                    self.testing_state[individual[0], individual[1]] = 3 # Person now positive!
                    
                    # Change the status to removed for positive cases in 
                    # isolation to keep the count right
                    self.disease_state[individual[0], individual[1]] = 4
            
                # The person is negative
                else:
                                    
                    self.testing_state[individual[0],individual[1]] = 0 # Test status reset
                    self.test_results[individual[0],individual[1]]  = 0 # Test results reset
                            
            # Advance the testing clock
            self.test_clock[individual[0], individual[1]] += 1
                
        # ------ Exposed people ------
    
        exposed = np.argwhere(self.disease_state == 1)

        for individual in exposed:
            
            advance_disease(individual)
                
        # ------ Advance infection clock for infectious suspects ------
    
        isolated_infectious = np.argwhere(
            np.logical_and(np.logical_or(self.disease_state == 2, 
                                         self.disease_state == 3), 
                           self.testing_state != 0)
            )

        for individual in isolated_infectious:

            advance_disease(individual)

        self.time += 1


    def simulate(self, NTime = 1000, num_initial_infectious = 3, verbose = 0):     
        """Simulate an epidemic.

        Parameters
        ----------

        NTime : int
            Number of simulation time steps
        num_initial_infectious : int 
            Number of initial weak-symptom infectious
        verbose : Verbose output if 1

        Returns
        -------
        population : numpy.array
            [NTime, 6] array where rows correspond to times and columns 
            correspond to states 0: S, 1: E, 2: I_W, 3: I_S, 4: I_R, 5: cases
       
        """        
        
        self.add_infectious(num_initial_infectious = num_initial_infectious)

        # Initialize population array
        population = np.zeros((NTime, 6))
        number_of_tests = np.zeros(NTime)

        # Advance the infection and testing in time
        while self.time < NTime:
                    
            population[self.time, 0] = np.sum(self.disease_state==0) # S
            population[self.time, 1] = np.sum(self.disease_state==1) # E
            population[self.time, 2] = np.sum(self.disease_state==2) # I_w
            population[self.time, 3] = np.sum(self.disease_state==3) # I_s
            population[self.time, 4] = np.sum(self.disease_state==4) # I_R
            population[self.time, 5] = (np.sum(self.testing_state==3) 
                                      + np.sum(self.testing_state==4)) # cases
            number_of_tests[self.time] = self.tests_today

            # Display some information about the simulation
            if self.time%10 == 0 and verbose:
                print(
         f'simulation step: {self.time:4d}, '
        +f'active infections: {np.int(population[self.time,2]+population[self.time,3]):7d},'
        +f'recovered: {np.int(population[self.time,4]):7d}, '
        +f'tested positive: {np.int(population[self.time,5]):7d}'
                    )
                
            # Stop if no susceptible left
            if population[self.time, 0] == 0:
                
                if verbose:
                    
                    print("Everyone is infected")
                
                break
            
            # Stop if no infectious
            
            if (population[self.time, 1] == 0 
            and population[self.time, 2] == 0  
            and population[self.time, 3] == 0):
                
                if verbose:
                    
                    print("No active infection")
                
                break
            
            # Take a time step
            self.evolve()
        
        return population, number_of_tests
    
    
    def reproduction_number(self):
        """
        Computes and returns an approximation to the basic reproduction number 
        at model parameters. 
        """
        
        gamma_I = self.kIR * self.thetaIR # Mean infectious time

        # Contribution of the random interactions
        term_random = gamma_I * (self.prob_symptom * self.etaS 
                            + (1 - self.prob_symptom) * self.etaW)

        # Contribution of the nearest neighbor interaction
        term_neighbor = (self.prob_symptom * (1 - (1 - self.tauS) ** gamma_I)  
                    + (1 - self.prob_symptom) * (1 - (1 - self.tauW) ** gamma_I)
                    )


        R_fun = lambda z: term_random + z * term_neighbor
        z_fun = lambda R, zz: 4 * term_random / R + 2.5 * zz * term_neighbor / R
        
        z_guess = 4
        R_guess = R_fun(z_guess)

        z_new = z_fun(R_guess, z_guess)
        R_new = R_fun(z_new)

        while abs((z_new - z_guess) / z_new 
                + (R_new - R_guess) / R_new) > 1e-3:

            z_guess = z_new
            R_guess = R_new

            z_new = z_fun(R_guess, z_guess)
            R_new = R_fun(z_new)

        R_zero = R_new

        return R_zero 