def convolution_matrix(src, dst, kernel, toric=False):
    '''
    Build a sparse convolution matrix M such that:

    (M*src.ravel()).reshape(src.shape) = convolve2d(src,kernel)

    You can specify whether convolution is toric or not and specify a different
    output shape. If output (dst) is different, convolution is only applied at
    corresponding normalized location within the src array.

    Building the matrix can be pretty long if your kernel is big but it can
    nonetheless saves you some time if you need to apply several convolution
    compared to fft convolution (no need to go to the Fourier domain).

    Parameters:
    -----------

    src : n-dimensional numpy array
        Source shape

    dst : n-dimensional numpy array
        Destination shape

    kernel : n-dimensional numpy array
        Kernel to be used for convolution

    Returns:
    --------

    A sparse convolution matrix

    Examples:
    ---------

    >>> Z = np.ones((3,3))
    >>> M = convolution_matrix(Z,Z,Z,True)
    >>> print (M*Z.ravel()).reshape(Z.shape)
    [[ 9.  9.  9.]
     [ 9.  9.  9.]
     [ 9.  9.  9.]]
    >>> M = convolution_matrix(Z,Z,Z,False)
    >>> print (M*Z.ravel()).reshape(Z.shape)
    [[ 4.  6.  4.]
     [ 6.  9.  6.]
     [ 4.  6.  4.]]
    '''
 
    # For a toric connection, it is wrong to have a kernel larger
    # than the source
 #   if toric:
 #       shape = np.minimum(np.array(src.shape), np.array(kernel.shape))
 #       kernel = extract(kernel, shape, np.rint(np.array(kernel.shape)/2.))

    # Get non NaN value from kernel and their indices.
    nz = (1 - np.isnan(kernel)).nonzero()
    data = kernel[nz].ravel()
    indices = [0,]*(len(kernel.shape)+1)
    indices[0] = np.array(nz)
    indices[0] += np.atleast_2d((np.array(src.shape)//2 - np.array(kernel.shape)//2)).T

    # Generate an array A for a given shape such that given an index tuple I,
    # we can translate into a flat index F = (I*A).sum()
    to_flat_index = np.ones((len(src.shape),1), dtype=int)
    if len(src.shape) > 1:
        to_flat_index[:-1] = src.shape[1]

    R, C, D = [], [], []
    dst_index = 0
    src_indices = []

    # Translate target tuple indices into source tuple indices taking care of
    # possible scaling (this is done by normalizing indices)
    for i in range(len(src.shape)):
        z = np.rint((np.linspace(0,1,dst.shape[i])*(src.shape[i]-1))).astype(int)
        src_indices.append(z)

    nd = [0,]*(len(kernel.shape))
    for index in np.ndindex(dst.shape):
        dims = []
        # Are we starting a new dimension ?
        if index[-1] == 0:
            for i in range(len(index)-1,0,-1):
                if index[i]: break
                dims.insert(0,i-1)
        dims.append(len(dst.shape)-1)
        for dim in dims:
            i = index[dim]

            if toric:
                z = (indices[dim][dim] - src.shape[dim]//2 +(kernel.shape[dim]+1)%2 + src_indices[dim][i]) % src.shape[dim]
            else:
                z = (indices[dim][dim] - src.shape[dim]//2 +(kernel.shape[dim]+1)%2 + src_indices[dim][i])

            # if toric:
            #     z = (indices[dim][dim] - src.shape[dim]/2.0 -(kernel.shape[dim]+1)%2 + src_indices[dim][i]) % src.shape[dim]
            # else:
            #     z = (indices[dim][dim] - src.shape[dim]/2.0 -(kernel.shape[dim]+1)%2+ src_indices[dim][i])

            n = np.where((z >= 0)*(z < src.shape[dim]))[0]
            if dim == 0:
                nd[dim] = n.copy()
            else:
                nd[dim] = nd[dim-1][n]
            indices[dim+1] = np.take(indices[dim], n, 1)
            indices[dim+1][dim] = z[n]
        dim = len(dst.shape)-1
        z = indices[dim+1]
        R.extend( [dst_index,]*len(z[0]) )
        C.extend( (z*to_flat_index).sum(0).tolist() )
        D.extend( data[nd[-1]].tolist() )
        dst_index += 1

    return sp.coo_matrix( (D,(R,C)), (dst.size,src.size))
