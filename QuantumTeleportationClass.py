import numpy as np
import sys
from random import choices
import random 


class QuantumMethods:
    
    def __init__(self):
        self.self = self

    def randomness(self):
        '''
        
        This function is used to generate a random initial state for the quantum
        teleportation process to use.
        
        
        Returns
        -------
        alpha : complex pre-factor of the |0> state
        beta : complex pre-factor of the |1> state
        func : the initial state, that has been formed from random number generation

        '''
        
        a, b, c, d =  np.random.uniform(-1, 1, 4)  
        
        
        alpha2 = a**2 + c**2
        
        beta2 = b**2 + d**2
        
        norm = np.sqrt(alpha2 + beta2)
        
        alpha = (a + c*1j)/norm
        beta = (b + d*1j)/norm
        
        
        assert abs(alpha)**2 + abs(beta)**2 >= 0.9999999
        
        first, second = self.basis_vectors()
        
        func = alpha*first + beta*second
        
        return  alpha, beta, func
    
    
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
    
    
    
    def tensor_prod(self, *args):
        """
        Parameters
        ----------
        *args: N-matrices (array-like) of which the tensor product will be taken. N must be > 1.
      
        Returns
        -------
        result : Tensor Product of the input matrices (array-like) of shape.
        
        """
        #print(args[0])
        result = args[0]
        

        for i, tensors in enumerate(args):
            if i == 0:
                continue

            n, m = np.shape(result)
            q, p = np.shape(tensors)

            product = np.empty((q*n, m*p), dtype=complex)
            for i in range(0, np.shape(product)[0]):
                for j in range(0, np.shape(product)[1]):
                  
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
        assert np.shape(t1)[1] == np.shape(t2)[0] # assert shapes satisfy standard matrix calc requirement
        t1_shape = np.shape(t1)
        t2_shape = np.shape(t2)
        # set shape of output
        fin = np.ones((t1_shape[0], t2_shape[1]), dtype = complex)
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
                gate = self.tensor_prod(gate, base) 
        
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
        elif gateType == 'IDENTITY': 
            gate = np.array([[1,0],[0,1]])
            
        elif gateType == 'X':  # Pauli-X
            gate = np.array([[0, 1], [1, 0]])
            
        elif gateType == 'Z':  # Pauli-Z
            gate = np.array([[1, 0], [0, -1]])
              
        else:
            #print("Gate type '" + gateType + "' not recognised.")
            sys.exit()
        
        #print('target: ' , target)
        #print(gate, '<- gate')
        
        assert len(gate) == len(target)
   
        new = self.matrix_prod(gate, target)  #this is applying the gate to the register
        
        #print('result: ' , new)
        
        return new
    
    
    def applying_to_1_qubit(self, n, t):
        '''
        
        This function allows for gates to be applied to individually specified 
        qubits within a register.
        
        n: number of qubits 
        t: target qubit for the gate 

        '''
        Identity = np.array([[1,0],[0,1]])
        Had = np.array([[1,1],[1,-1]]) / np.sqrt(2)
        CNOT = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]])
        X = np.array([[0, 1], [1, 0]])
        Z= np.array([[1, 0], [0, -1]])
        #Y = np.array([[0, -1j], [1j, 0]])
        
        
        matrices = np.array([X, Z], dtype=object)
        
        input_to_tensor = []
        
        if t == 1:
            for i in range(1, n):
                if i == t:
                    input_to_tensor.append(CNOT)
                else:
                    input_to_tensor.append(Identity)
                    trythis = self.tensor_prod(*input_to_tensor)
        elif t == 0:
            for i in range(0, n):
                if i == t:
                    input_to_tensor.append(Had)
                else:
                    input_to_tensor.append(Identity)
                    trythis = self.tensor_prod(*input_to_tensor)
    
            
        elif t == 3:
            input_to_tensor.append(X)
            input_to_tensor.append(Identity)
            input_to_tensor.append(Identity)
            trythis = self.tensor_prod(*input_to_tensor)
        elif t == 4:
            input_to_tensor.append(Z)
            input_to_tensor.append(Identity)
            input_to_tensor.append(Identity)
            trythis = self.tensor_prod(*input_to_tensor)
            
        elif t ==5: 
            input_to_tensor.append(Z)
            input_to_tensor.append(Identity)
            input_to_tensor.append(Identity)
            trythis = self.tensor_prod(*input_to_tensor)
            
        elif t ==6: 
            rand_choice = random.choice([X, Z])
            input_to_tensor.append(rand_choice)
            rand_choice = random.choice([X, Z])
            input_to_tensor.append(rand_choice)
            rand_choice = random.choice([X, Z])
            input_to_tensor.append(rand_choice)
            first = self.tensor_prod(*input_to_tensor)
            trythis = first 
        
        elif t == 10:
            rand_choice = random.choice([X, Z])
            trythis = rand_choice
            
        return trythis
                
    
    
    
    def randMeasure(self, n , register):
        '''
        This function observes the probability of measurement of a qubit and uses
        that to collapse the wavefunction.

        Parameters
        ----------
        n : The qubit that is to be measured
        register : the register that contains the qubit to be measured

        Returns
        -------
        register_new : the collapsed register after the measurments
        flag : returns either 0 or 1 depending on the probability of measurement

        '''
        
        if n == 2:
           p0 = abs(register[0])**2 + abs(register[1])**2 + abs(register[4])**2 + abs(register[5])**2
           p1 = abs(register[2])**2 + abs(register[3])**2 + abs(register[6])**2 + abs(register[7])**2
           
           flag = choices([0,1], [p0, p1])[0]
         
           
           if flag == 0:
               register_new = 1/np.sqrt(p0)*[register[0], register[1], register[4], register[5]]
           else:
               register_new = 1/np.sqrt(p1)*[register[2], register[3], register[6], register[7]]
        
        
        if n == 1:
            p0 = abs(register[0])**2 + abs(register[1])**2
            p1 = abs(register[2])**2 + abs(register[3])**2
            
            flag = choices([0,1], [p0, p1])[0]
          
            
            if flag == 0:
                register_new = 1/np.sqrt(p0)*[register[0], register[1]]
            else:
                register_new = 1/np.sqrt(p1)*[register[2], register[3]]
         
            
        return register_new , flag
    
    
    def FinalStep(self, measurement_qubit_1, measurment_qubit_2, new_new_reg):
        '''
        This function applies the final corrections to qubit 3 depending on the classical
        measurements determined in the randMeasure function

        Parameters
        ----------
        measurement_qubit_1 : TYPE
            DESCRIPTION.
        measurment_qubit_2 : TYPE
            DESCRIPTION.
        new_new_reg : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        
        if measurement_qubit_1 == 0 and measurment_qubit_2 == 0:
            return new_new_reg
        
        elif measurement_qubit_1 == 0 and measurment_qubit_2==1:
            new = self.gate('X', new_new_reg, 1)
            return new
        
        elif measurement_qubit_1 == 1 and measurment_qubit_2==0:
            new = self.gate('Z', new_new_reg, 1)
            return new
        else:
            new = self.gate('X', new_new_reg, 1)
            new_new = self.gate('Z', new, 1)
            return new_new
        
        
    
    def applyGATENoise(self, register, p, g):
        '''
        

        Parameters
        ----------
        register : the quanutm register where noise is applied
        p : the probability of noise being applied
        g : special case of when the moise is applied to a 1 qubit register

        Returns
        -------
        Returns a register or qubit that has had noise applied if a randomly generated number is less than p

        '''
        
        
        Identity = np.array([[1,0],[0,1]])
        Identity3 = [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]]
        
        q = random.uniform(0, 1)
       
        if q < p:
            if g != 7:
                    choice = random.choice([1, 2, 3])
                    choice = 4
                    
                    if choice == 1:
                        
                        MAT = self.applying_to_1_qubit(3, 3)
                        noise = self.matrix_prod(MAT, register)
                    if choice == 2:
                       
                        MAT = self.applying_to_1_qubit(3, 4)
                       
                        noise = self.matrix_prod(MAT, register)
                    if choice == 3:
                       
                        MAT = self.applying_to_1_qubit(3, 5)
                        noise = self.matrix_prod(MAT, register)
                        
                    if choice == 4:
                       
                        MAT = np.array(self.applying_to_1_qubit(3, 6)) * (p / 3) + (1 - p) * np.array(Identity3)

                        noise = self.matrix_prod(MAT, register)
                        
                
            elif g == 7:
                MAT = self.applying_to_1_qubit(1, 10) * p/3 + (1-p)*Identity
                noise = self.matrix_prod(MAT, register)
                
            return noise
        else:
            return register
    
    
        
    

    def teleportation_noise(self, n, vector_used, second_vector_used, p):   #vector used us |0>, second vector used is |1> 
        
        list_of_vectors = [vector_used] * n 
         
        list_of_1_vectors = [second_vector_used] *n
        
        
        
        state1 = self.tensor_product_register(*list_of_vectors)     
        state3 = self.tensor_product_register(*list_of_1_vectors)   
        
       
        
        bell = (1/np.sqrt(2)) *  (state1 + state3)
     
        alpha, beta, state2 = self.randomness()
      
        register = self.tensor_prod(state2, bell)   
       
       
        CNOT_apply = self.applying_to_1_qubit(3, 1)    
       
        result_of_CNOT_1 = self.matrix_prod(CNOT_apply, register)  
        
        result_of_CNOT = self.applyGATENoise(result_of_CNOT_1, p, 8)
       
        
        Had_apply = self.applying_to_1_qubit(3, 0)  
        
       
        
        result_of_HAD_1 = self.matrix_prod(Had_apply, result_of_CNOT) 
        
        result_of_HAD = self.applyGATENoise(result_of_HAD_1, p, 8)
        
        new_reg, measurment_qubit_2 = self.randMeasure(2, result_of_HAD) 
        
        new_new_reg, measurement_qubit_1 = self.randMeasure(1, new_reg)
        
        print('This is qubit 1: ' + '\n' +  str(state2))
        
        print('The chance of noise is: ' + str(p))
        
        print('This is qubit 3 pre-correction:' + '\n' + str(new_new_reg))
        print('This is the outcome of measurement: ' + '\n' + str(measurement_qubit_1) + str(measurment_qubit_2))
    
        print('This is qubit 3 post correction: ' + '\n' +str(self.FinalStep(measurement_qubit_1, measurment_qubit_2, new_new_reg))) 
      
        original_state = state2
        teleported_state = self.FinalStep(measurement_qubit_1, measurment_qubit_2, new_new_reg)
       
        
        return original_state, teleported_state 
        
    
    
    def teleport_no_noise(self, n, vector_used, second_vector_used):  
        
        
        '''
        
        This function runs the quantum telepotation algorithm with no simulated noise

        Parameters
        ----------
        n : n = (number of qubits - 1) of the total system 
        vector_used : computational basis vector |0>
        second_vector_used : computational basis vector |1>

        Returns
        -------
        original_state : The state that was initialised as qubit 1 at the start
        teleported_state : The state that 

        '''
        
        list_of_vectors = [vector_used] * n 
         
        list_of_1_vectors = [second_vector_used] *n
        

        state1 = self.tensor_product_register(*list_of_vectors)      #|00> state
        state3 = self.tensor_product_register(*list_of_1_vectors)    #|11> state
        
        
        bell = (1/np.sqrt(2)) *  (state1 + state3)
        
        alpha, beta, state2 = self.randomness()
        
        print('This is qubit 1 initially:'   + '\n' + str(state2))
        
        register = self.tensor_prod(state2, bell)
       
        
        CNOT_apply = self.applying_to_1_qubit(3, 1)
        
        
        result_of_CNOT = self.matrix_prod(CNOT_apply, register)
        
        
        Had_apply = self.applying_to_1_qubit(3, 0)
        
        
        result_of_HAD = self.matrix_prod(Had_apply, result_of_CNOT)
        
        new_reg, measurment_qubit_2 = self.randMeasure(2, result_of_HAD) #measuring 2 
        
        new_new_reg, measurement_qubit_1 = self.randMeasure(1, new_reg)
        
        
        print('This is the outcome of measurement: ' + str(measurement_qubit_1) + str(measurment_qubit_2))

        print('This is qubit 3 before correction: ' + '\n' +  str(new_new_reg))
        
        original_state = state2
        
        teleported_state = self.FinalStep(measurement_qubit_1, measurment_qubit_2, new_new_reg)
       
        print('This is qubit 3 after correction: ' + '\n' +  str(teleported_state))
        
        return   original_state, teleported_state   
    
    
    


    
    
        
        
        
    
    
