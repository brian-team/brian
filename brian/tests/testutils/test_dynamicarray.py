from numpy import array

from brian.utils.dynamicarray import DynamicArray, DynamicArray1D

def test_dynamicarray():
    
    # Perform the test mentioned in the docstring of the DynamicArray class.
    # TODO: This could be a doctest directly... 
    x = DynamicArray((2, 3), dtype=int)
    x[:] = 1
    x.resize((3, 3))
    x[:] += 1
    x.resize((3, 4))
    x[:] += 1
    x.resize((4, 4))
    x[:] += 1
    x.data[:] = x.data**2
    
    # Do a resize that changes nothing
    x.resize((4, 4))
    
    # This is the expected result
    y = array([[16, 16, 16, 4],
               [16, 16, 16, 4],
               [ 9,  9,  9, 4],
               [ 1,  1,  1, 1]])
    
    assert(len(x) == len(y))
    assert(len(x[0]) == len(y[0]))
    assert(x.shape == (4, 4))    
    assert((x == y).all())

    # Do the same with use_numpy_resize=True
    x = DynamicArray((2, 3), dtype=int, use_numpy_resize=True)
    x[:] = 1
    x.resize((3, 3))
    x[:] += 1
    x.resize((3, 4))
    x[:] += 1
    x.resize((4, 4))
    x[:] += 1
    x.data[:] = x.data**2
    
    # Do a resize that changes nothing
    x.resize((4, 4))
    
    y = array([[16, 16, 16, 4],
               [16, 16, 16, 4],
               [ 9,  9,  9, 4],
               [ 1,  1,  1, 1]])
    
    assert(len(x) == len(y))
    assert(len(x[0]) == len(y[0]))
    assert(x.shape == (4, 4))    
    assert((x == y).all())
    
    # Test shrinking
    x = DynamicArray((2, 3), dtype=int)
    x[:] = 1
    x.shrink((1, 2))
    # This should not do anything
    x.shrink((2, 3))
    y = array([[1, 1]])

    assert(x.shape == y.shape)
    assert((x == y).all())
        
    # Test DynamicArray1D
    x = DynamicArray1D(2, dtype=int)
    x[:] = 1
    x.resize(3)
    x[:] += 1
    x.resize(4)
    x.data[:] = x.data**2

    # Expected result
    y = array([4, 4, 1, 0])
    
    assert(len(x) == len(y))
    assert(x.shape == y.shape)
    assert((x == y).all())
    
    # Test DynamicArray1D with use_numpy_resize=True
    x = DynamicArray1D(2, dtype=int, use_numpy_resize=True)
    x[:] = 1
    x.resize(3)
    x[:] += 1
    x.resize(4)
    x.data[:] = x.data**2

    # Expected result
    y = array([4, 4, 1, 0])
    
    assert(len(x) == len(y))
    assert(x.shape == y.shape)
    assert((x == y).all())

if __name__ == '__main__':
    test_dynamicarray()