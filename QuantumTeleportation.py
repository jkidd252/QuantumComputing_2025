import numpy as np
from QuantumTeleportationClass import QuantumMethods
import matplotlib.pyplot as plt


QM = QuantumMethods()
first, second = QM.basis_vectors()


class main():
    '''
    This is the main function that runs the no noise, and noise implementations of 
    quantum teleportation.
    
    To run an implementation the following options are available:
    
        no_noise() = just the basic no noise implementationof quantum teleportation 
        noise() = the basic noise implementation of quantum teleportation    
        noise_line_graph(p) = plots the line showing the difference of original and teleproted state
            - here p is the probability of noise 
        noise_fid_grap() = plots the fidelity of a noisy system as the noise of the system increases 
        
    
    '''
    
    
    def no_noise():
        '''
        This function runs just the basic no noise quantum teleportation
        '''
        
        print('THIS IS A NON-NOISE IMPLEMENTATION:')
        
        original_state, teleported_state = QM.teleport_no_noise(2, first, second)


    def noise():
        '''
        This function runs the noise version of quantum teleportation
        '''
        
        print('THIS IS A NOISE IMPLEMENTATION:')
        
        original_state, teleported_state = QM.teleportation_noise(2, first, second, 0.9)

        
    def noise_line_graph(p):
        '''
        A  function that plots the graph showing the loss of information based on the noise supplied 

        '''
        
        state0 = []
        state1 = []
        fidelity = []
    
        final_fid = []
        FID_STD_lower = []
        FID_STD_upper = []
        prob = []


        for i in range(0, 1000):
            original_state, teleported_state = QM.teleportation_noise(2, first, second, p)
            state0.append(abs(original_state[0]))
            state1.append(abs(teleported_state[0]))
    
    
        plt.figure(figsize = (5, 5))
        plt.plot(state0, state1, marker = 'x', ls = 'none', color = 'black')
        plt.xlabel(r'$|\alpha|_{original}$')
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.ylabel(r'$|\alpha|_{teleported}$')
        plt.savefig('90Per.png', dpi = 1000)
        
    def noise_fid_graph():
        
        '''
        A  function that plots the graph showing the fidelity loss against increasing noise 

        '''
        
        final_fid = []
        FID_STD_upper = []
        FID_STD_lower =[]
        prob = []
        
        state0 = []
        state1 = []
        
        for p in range(0, 100):
            fidelity = []
            for i in range(0, 50):
                original_state, teleported_state = QM.teleportation_noise(2, first, second, p/100)
                fidelity_1 = (np.abs(np.vdot(original_state, teleported_state))**2)
                fidelity.append(fidelity_1)
        

                state0.append((original_state[0]))
                state1.append((teleported_state[0]))
                
            final_fid.append(np.mean(fidelity))
            lower_error = np.std(fidelity)  
            upper_error = min(1 - np.mean(fidelity), np.std(fidelity))
            FID_STD_lower.append(lower_error)
            FID_STD_upper.append(upper_error)

            
            
            prob.append(p/100)

        plt.figure(figsize = (5,5))
        plt.show()
        plt.errorbar(prob,  final_fid, yerr=[FID_STD_lower, FID_STD_upper],   marker = 'x', ls = 'none', capsize=(2), color = 'black')
        plt.ylabel('Fidelity of system')
        plt.xlabel('Probability of Noise when applying Gates')
        plt.ylim(0, 1.02)
        plt.xlim(0,1)
        plt.axhline(y = 0.5, color = 'red', ls = '--')
        plt.savefig('FidAgainst.png', dpi = 1000)

