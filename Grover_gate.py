import numpy as np
from QuantumMethodsClass import QuantumMethods
from tensor_prod import matrix_prod, tensor_prod
from oracle import generate_oracle, generate_diffuser


#def hadamard(n):
   # """Returns an n-qubit Hadamard matrix."""
   # H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
   # Hn = H
  #  for _ in range(n - 1):
   #     Hn = QM.tensor_prod(Hn, H)
   # return Hn

def grover_Gate(n, target, iterations, vector_used, QM):
    
    list_of_vectors = [vector_used] * n  
    
    """Simulates Grover's algorithm for n qubits, searching for 'target' state."""
    
    #state1 = QM.tensor_prod(*list_of_vectors)
    state1 = tensor_prod(*list_of_vectors)
    #print(type(state1))
    
    UniformedHad = QM.gate('H', state1, 0)
    #print('This: ' + str(UniformedHad))
    state = UniformedHad
    

    O = generate_oracle(n, target)
    #print(np.shape(O))
    D = generate_diffuser(n)

    #print("Oracle matrix:\n", O)  # Debugging statement
    #print("Diffuser matrix:\n", D)  # Debugging statement

    for i in range(iterations):
        state = matrix_prod(O,state)                         # Apply Oracle
        #print(f"After Oracle {i + 1}: {np.round(state, 4)}")    # Debugging
        state = matrix_prod(D,state)  # Apply Diffusion
        #print(f"After Diffusion {i + 1}: {np.round(state, 4)}") # Debugging

    return np.abs(state) ** 2  # Return probability distribution
