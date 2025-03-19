import numpy as np
from QuantumMethodsClass import QuantumMethods
from tensor_prod import tensor_prod
from matplotlib import pyplot as plt


#def hadamard(n):
   # """Returns an n-qubit Hadamard matrix."""
   # H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
   # Hn = H
  #  for _ in range(n - 1):
   #     Hn = QM.tensor_prod(Hn, H)
   # return Hn


def oracle(n, target):
  """Creates an oracle matrix that flips the phase of the target state."""
  N = 2 ** n
  O = np.eye(N)
  O[target, target] = -1  # Phase flip for the marked state
  return O


def diffuser(n):
 """Creates the Grover diffusion matrix.""" 
 N = 2**n
 S = 2 * np.ones((N, N)) / N - np.eye(N)  # Correct mean inversion
 return S



def grover(n, target, iterations, vector_used, QM):
    
    list_of_vectors = [vector_used] * n  
    
    """Simulates Grover's algorithm for n qubits, searching for 'target' state."""
    
    #state1 = QM.tensor_prod(*list_of_vectors)
    state1 = tensor_prod(*list_of_vectors)
    #print(type(state1))
    
    UniformedHad = QM.gate('H', state1, 0)
    #print('This: ' + str(UniformedHad))
    state = UniformedHad
    

    O = oracle(n, target)
    #print(np.shape(O))
    D = diffuser(n)

    #iteration = []
    #sparse_percentage = []
    for i in range(iterations):
        state = QM.matrix_prod(O,state)                         # Apply Oracle
        #print(f"After Oracle {i + 1}: {np.round(state, 4)}")    # Debugging
        state = QM.matrix_prod(D,state)  # Apply Diffusion
        #print(f"After Diffusion {i + 1}: {np.round(state, 4)}") # Debugging
        #num_zeros = np.count_nonzero(state)
        #sparse_perc = ((np.shape(state)[0] * np.shape(state)[1]) - num_zeros) / (np.shape(state)[0] * np.shape(state)[1]) 
        #sparse_percentage.append(sparse_perc)
    
    #plt.plot(iteration, sparse_percentage)
    #plt.xlabel('Number of Iterations')
    #plt.ylabel('Sparsity, %')
    #plt.savefig('spars_vs_iter.png', dpi=1000)
    #plt.show()
    return np.abs(state) ** 2  # Return probability distribution
