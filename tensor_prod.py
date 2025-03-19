import numpy as np

def matrix_prod(*args):
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
    t1 = args[0]
    
    for i, t2 in enumerate(args):
        if i == 0:
            continue
            
        assert np.shape(t1)[1] == np.shape(t2)[0] # assert shapes satisfy standard matrix calc requirement
        t1_shape = np.shape(t1)
        t2_shape = np.shape(t2)
        # set shape of output
        fin = np.ones((t1_shape[0], t2_shape[1]))
        # iterator - use inner index to iterate over elements within matrices
        k = np.arange( t1_shape[1] )

        for i in range(0, t1_shape[1]):
            for j in range(0, t2_shape[1]):
                # edit element [i,j] within output matrix
                fin[i, j] = np.sum( t1[i, k] * t2[k, j] )

    return fin   
    
    
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

    for i, tensors in enumerate(args):
        if i == 0:
            continue

        n, m = np.shape(result)
        q, p = np.shape(tensors)

        product = np.empty((q*n, p*m))

        for i in range(0, np.shape(product)[0]):
            for j in range(0, np.shape(product)[1]):
                product[i,j] = result[ i//q, j//p ]* tensors[ i%q, j%p ]
        result = product
    return result