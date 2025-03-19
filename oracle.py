import numpy as np
from tensor_prod import matrix_prod, tensor_prod


I = np.eye(2)
X = np.array([[0,1],[1,0]])
H = np.array([[1,1],[1,-1]]) / np.sqrt(2)

def generate_oracle(n,target):
    """
    Generates the oracle for Grover's algorithm

    Parameters
    ----------
    n : integer
        number of qubits in the register
    target : integer
        target state position in the register

    Returns
    -------
    oracle : 2^n by 2^n array of floats
        matrix form of the grover's algorithm oracle for the target state
        [note: CtothenminusoneZ is the oracle for the last state in the register] 

    """
    CtothenminusoneZ = np.eye(2**n)
    CtothenminusoneZ[-1,-1] = -1
    
    
    if target == 2**n-1:
        return CtothenminusoneZ
    
    NOT = np.array([[1]])
    for i in range(n):
        if target & (1 << i):
            NOT = tensor_prod(I, NOT)
        else:
            NOT = tensor_prod(X, NOT)

    oracle = matrix_prod(NOT, CtothenminusoneZ)
    oracle = matrix_prod(oracle, NOT)
    return oracle        


def generate_diffuser(n):
    """
    Generates the diffuser for Grover's algorithm

    Parameters
    ----------
    n : integer
        number of qubits in the register

    Returns
    -------
    diffuser : 2^n by 2^n array of floats
        matrix form of the Grover's algorithm diffuser

    """
    
    CtotheminusoneNOT = np.eye(2**n)  
    CtotheminusoneNOT[-1,-1] = 0
    CtotheminusoneNOT[-1,-2] = 1
    CtotheminusoneNOT[-2,-1] = 1
    CtotheminusoneNOT[-2,-2] = 0
    
    
    HCs = I
    for i in range(0,n-1):
        HCs = tensor_prod(H,HCs)
    
    diffuser = matrix_prod(HCs, CtotheminusoneNOT)
    diffuser = matrix_prod(diffuser, HCs)
    return diffuser