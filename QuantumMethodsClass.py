import numpy as np
import sys
from tensor_prod import tensor_prod, matrix_prod

class QuantumMethods:
    
    def __init__(self):
        self.self = self

    def basis_vectors(self):
        
        '''
        This sets up the computational basis vectors
        Parameters
        ----------
        N/A

        Returns
        -------
        The basis state vectors |0> and |1>

        '''

        base_0 = np.array([[float(1)], [float(0)]])
        base_1 = np.array([[float(0)], [float(1)]])
        
        return base_0, base_1
    
    
    def tensor_product_register(self, *args):
        
        '''
        This forms a register based on a predefined number of qubits (i.e. a 2 qubit system)
        Parameters
        ----------
        *args : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        #print(len(args))
        if len(args) < 2:
            X = 'NOT ENOUGH QUBITS TO FORM A REGISTER, PLEASE ENTER AT LEAST 2'
            return X
        else:
            result = args[1]
        
            for i, states in enumerate(args):
            
                if i == 1:
                    continue
    
                first = states[0]*result
                second = states[1]*result
                result = np.concatenate((first, second))
        
            return result
    
    
    
    def tensor_prod(*args):
        """
        Parameters
        ----------
        *args: N-matrices (array-like) of which the tensor product will be taken. N must be > 1.
      
        Returns
        -------
        result : Tensor Product of the input matrices (array-like) of shape.
        """
        result = args[0]
        #print(result)

        for i, tensors in enumerate(args):
            if i == 0:
                continue
                print(tensors)
                n, m = np.shape(result)
                q, p = np.shape(tensors)

                product = np.empty((q*n, p*m))
                #print(np.shape(product)[0])
                #print(np.shape(product)[1])

                for i in range(0, np.shape(product)[0]):
                    #print(i)
                    for j in range(0, np.shape(product)[1]):
                        #print(j)
                        product[i,j] = result[ i//q, j//p ]* tensors[ i%q, j%p ]
                result = product
        return result
    
    
    def matrix_prod(self, t1, t2):
        """
        Parameters
        ----------
        t1, t2 : Two matrices (array-like), of the same shape, of which the product will be taken 

        Their shapes must match along the inner axis, i.e. A_{ij}*B_{nk} where j=n with output of shape (i,k)  
        This condition is controlled using an assertion.
            
        Returns
        -------
        fin : Product of two input matrices (array-like) of shape (i,k).
        """
        #print('This is T1: '  + str(t1))
        #print('This is T2: '  + str(t2))
        
        assert np.shape(t1)[1] == np.shape(t2)[0] # assert shapes satisfy standard matrix calc requirement
        
        t1_shape = np.shape(t1)
        #print(t1_shape[1])
        
        t2_shape = np.shape(t2)
        #print(t2_shape[0])
        # set shape of output
        
        fin = np.ones((t1_shape[0], t2_shape[1]))
        
        
        # iterator - use inner index to iterate over elements within matrices
        k = np.arange( t1_shape[1] )

        for i in range(0, t1_shape[1]):
            for j in range(0, t2_shape[1]):
                # edit element [i,j] within output matrix
                fin[i, j] = np.sum( t1[i, k] * t2[k, j] )

        return fin   
    
    
    '''
    This is a class that contains all of the functions and methods used in running
    the baseline version of our grovers algorithm simulation
    '''
 
    
    def gate(self, gateType, target, arg):
        """
        Applies a quantum logic gate to a qubit register.
    
        Parameters
        ----------
        gateType : str
            Specifies which gate. Options:
                'H'
                'PHI'
                'CNOT'
                'CCNOT'
                'SWAP'
        target : 1-D numpy array
            qubit register
        arg : varies
            an extra argument for certain gates e.g. angle for phase shift gate
    
        Returns
        -------
        new: 1-D numpy array
            the updated qubit register
        """    
        n = np.log2(len(target))
        assert n % 1 == 0
        n = int(n)
        #print('# of qubits: ' , n)
        
        if gateType == 'H': # HADAMARD
            base = np.array([[1,1],[1,-1]]) / np.sqrt(2)
            gate = base
            for i in range(n-1):
                gate = tensor_prod(gate, base) 
        
        elif gateType == 'PHI': # PHASE SHIFT
            base = np.array([[1,0],[0,np.exp(arg * 1j)]])
            gate = base
            for i in range(n-1):
                gate = self.tensor_prod(gate, base)
                
        elif gateType == 'CNOT': # CONTROLLED NOT
            gate = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
            
        elif gateType == 'CCNOT': # TOFFOLI
            gate = np.array([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1],[0,0,0,0,0,0,1,0]])
        
        elif gateType == 'SWAP': # SWAP
            gate = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])
            
        else:
            print("Gate type '" + gateType + "' not recognised.")
            sys.exit()
        
        #print('target: ' , target)
        #print(gate, '<- gate')
        
        #assert len(gate) == len(target)
   
        #new = self.matrix_prod(gate, target)  #this is applying the gate to the register
        new = matrix_prod(gate, target) 
        #print('result: ' , new)
        
        return new
    
    
