import numpy as np
from QuantumMethodsClass_sparse import QuantumMethods_sparse
from tensor_prod_sparse import tensor_prod_sparse, Lazy_matrix, OneD_to_TwoD, matrix_prod_sparse
from matplotlib import pyplot as plt


def oracle(n, target):
  """Creates an oracle matrix that flips the phase of the target state."""
  N = 2 ** n
  O = np.eye(N)
  O[target, target] = -1  # Phase flip for the marked state
  O_laz = Lazy_matrix(O)
  return O_laz.to_OneD()


def diffuser(n):
 """Creates the Grover diffusion matrix.""" 
 N = 2**n
 S = 2 * np.ones((N, N)) / N - np.eye(N)  # Correct mean inversion
 S_laz = Lazy_matrix(S)
 return S_laz.to_OneD()


def grover_sparse(n, target, iterations, vector_used, QM):
    
    list_of_vectors = [vector_used] * n
    #list_of_vectors = np.reshape( np.shape(list_of_vectors)[1], np.shape(list_of_vectors)[0] )
    #list_of_vectors = Lazy_matrix(list_of_vectors).to_OneD()
    
    """Simulates Grover's algorithm for n qubits, searching for 'target' state."""
    
    #state1 = QM.tensor_prod(*list_of_vectors)
    state1 = tensor_prod_sparse(*list_of_vectors)
    #print(type(state1))
    
    UniformedHad = QM.gate('H', state1, 0)
    #print('This: ' + str(UniformedHad))
    state = UniformedHad
    #print(state[0])

    

    O = oracle(n, target)
    #print(O[0])
    #print(O)
    D = diffuser(n)
    #print(D[0])


    #print("Oracle matrix:\n", O)  # Debugging statement
    #print("Diffuser matrix:\n", D)  # Debugging statement

    for i in range(iterations):
        state = matrix_prod_sparse(O,state)                         # Apply Oracle
        #print(f"After Oracle {i + 1}: {np.round(state, 4)}")    # Debugging
        state = matrix_prod_sparse(D,state)  # Apply Diffusion
        #print(f"After Diffusion {i + 1}: {np.round(state, 4)}") # Debugging
        #state = OneD_to_TwoD(state)
        
    return np.abs(OneD_to_TwoD(state)) ** 2  # Return probability distribution
