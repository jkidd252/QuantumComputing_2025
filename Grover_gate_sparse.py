import numpy as np
from QuantumMethodsClass_sparse import QuantumMethods_sparse as QuantumMethods
from tensor_prod_sparse import matrix_prod_sparse, tensor_prod_sparse
from oracle_sparse import generate_oracle, generate_diffuser


#def hadamard(n):
   # """Returns an n-qubit Hadamard matrix."""
   # H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
   # Hn = H
  #  for _ in range(n - 1):
   #     Hn = QM.tensor_prod(Hn, H)
   # return Hn

def grover_Gate_sparse(n, target, iterations, vector_used, QM):
    
    list_of_vectors = [vector_used] * n  
    
    """Simulates Grover's algorithm for n qubits, searching for 'target' state."""
    
    #state1 = QM.tensor_prod(*list_of_vectors)
    state1 = tensor_prod_sparse(*list_of_vectors)
    #print(type(state1))
    
    UniformedHad = QM.gate('H', state1, 0)
    #print('This: ' + str(UniformedHad))
    state = UniformedHad
    

    O = generate_oracle(n, target)
    #print(np.shape(O))
    D = generate_diffuser(n)

    for i in range(iterations):
        state = matrix_prod_sparse(O,state)                         # Apply Oracle
        #print(f"After Oracle {i + 1}: {np.round(state, 4)}")    # Debugging
        state = matrix_prod_sparse(D,state)  # Apply Diffusion
        #print(f"After Diffusion {i + 1}: {np.round(state, 4)}") # Debugging

    return np.abs(state) ** 2  # Return probability distribution
