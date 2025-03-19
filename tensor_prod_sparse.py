import numpy as np
from collections import defaultdict

class Lazy_matrix:
    def __init__(self, matrix):
        self.row, self.col = np.shape(matrix)
        self.matrix = matrix

    def to_OneD(self):
        """
        Parameters
        ----------
        self: Lazy_matrix objects (i.e a numpy array or matrix)
      
        Returns
        -------
        OneDim : 1D compression of the input. All elements that have value 0 are ignored. 
        Element 0 of OneDim contains the shape of the 2D matrix and all elements thereafter 
        are of the form (Value, index).
        Where index relates to the matrix coords using row = index/#rows , colum = index%#rows .
        """
        OneDim = []
        OneDim.append( (self.row, self.col) ) # store original size in 1st element of list
        for inds, val in np.ndenumerate(self.matrix):
            if val != 0:
                index = self.col * inds[0] + inds[1]
                OneDim.append((val, index))
        return OneDim


def OneD_to_TwoD(OneDim):
    """
    Parameters
    ----------
    OneDim: A sparse matrix (i.e output of Lazy_matrix.to_OneD()).
  
    Returns
    -------
    new_mat : 2D version of sparse matrix, all none specified values are set to 0.
    """
    m,n = OneDim[0]
    new_mat = np.zeros(shape=(m,n))
    for i in range(1, len(OneDim)):
        val, index = OneDim[i]
        row = index//n
        col = index%abs(n)
        new_mat[row, col] = val
    return new_mat


def tensor_prod_sparse(*args):
    """
    Parameters
    ----------
    *args: A series of OneDim objects (i.e output of Lazy_matrix.to_OneD()) of which the 
    kronecker product is computed.
  
    Returns
    -------
    prod : Kronecker product of the input 1D matrices.
    """
    result = args[0]
    n, m = result[0] 
    res_val = [x[0] for x in result[1:]]
    res_ind = [x[1] for x in result[1:]]
    
    for i, tensor in enumerate(args[1:]):
        q, p = tensor[0]
        tens_val = [x[0] for x in tensor[1:]]
        tens_ind = [x[1] for x in tensor[1:]]
        
        new_n, new_m = n * q, m * p
        # new 1D tensor, element 0 is the shape of the output
        prod = [[new_n, new_m]]
        
        
        for i_res, val_res in zip(res_ind, res_val):
            for i_tens, val_tens in zip(tens_ind, tens_val):
                # divmod(x, y) calculates both x//y and x%y and returns both 
                row_res, col_res = divmod(i_res, m)
                row_tens, col_tens = divmod(i_tens, p)
                
                new_index = (row_res * q + row_tens) * new_m + (col_res * p + col_tens)
                new_value = val_res * val_tens
                
                prod.append([new_value, new_index])
        
       
        res_val = [x[0] for x in prod[1:]]
        res_ind = [x[1] for x in prod[1:]]
        n, m = new_n, new_m
    
    return prod
    

def matrix_prod_sparse(*args):
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
    A = args[0]
    for i, B in enumerate(args):
        if i == 0:
            continue
         
        A_cols, A_rows = A[0]
        B_cols, B_rows = B[0]
        # Check matrix multiplication condition
        if B_cols != A_rows:
            raise ValueError("Matrix dimensions do not match for multiplication.")

        B_dict = defaultdict(list)
        for val, index in B[1:]:
            row = index // B_cols
            col = index % B_cols
            B_dict[row].append((col, val))  # Store as (column, value)

        C_dict = defaultdict(float)

        # Perform multiplication
        for a_val, a_index in A[1:]:
            a_row = a_index // A_cols
            a_col = a_index % A_cols

            # Check if A's column exists in B (sparse speedup)
            if a_col in B_dict:
                for b_col, b_val in B_dict[a_col]:
                    c_index = a_row * B_cols + b_col  # Compute index for result
                    C_dict[c_index] += a_val * b_val

        # Convert dictionary result to compressed format
        C = [(A_rows, B_cols)]  # Store shape
        for index, val in C_dict.items():
            if val != 0:
                C.append((val, index))

    return C
                
            
    