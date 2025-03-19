import numpy as np
from tensor_prod_sparse import matrix_prod_sparse, tensor_prod_sparse, Lazy_matrix


I = np.eye(2)
I = Lazy_matrix(I).to_OneD()
X = np.array([[0,1],[1,0]])
X = Lazy_matrix(X).to_OneD()
H = np.array([[1,1],[1,-1]]) / np.sqrt(2)
H = Lazy_matrix(H).to_OneD()

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
    CtothenminusoneZ = Lazy_matrix(CtothenminusoneZ).to_OneD()

    
    if target == 2**n-1:
        return CtothenminusoneZ
    
    NOT = np.array([[1]])
    NOT = Lazy_matrix(NOT).to_OneD()
    for i in range(n):
        if target & (1 << i):
            NOT = tensor_prod_sparse(I, NOT)
        else:
            NOT = tensor_prod_sparse(X, NOT)

    oracle = matrix_prod_sparse(NOT, CtothenminusoneZ)
    oracle = matrix_prod_sparse(oracle, NOT)
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
    CtotheminusoneNOT = Lazy_matrix(CtotheminusoneNOT).to_OneD()
    
    
    HCs = I
    for i in range(0,n-1):
        HCs = tensor_prod_sparse(H,HCs)
    
    diffuser = matrix_prod_sparse(HCs, CtotheminusoneNOT)
    diffuser = matrix_prod_sparse(diffuser, HCs)
    return diffuser