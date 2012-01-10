'''
This class defines data structures for the Synapses object.

** ConstructionSparseStateVector

The main one is the ConstructionSparseStateVector, look into it's doc
for details!

TODO:
- Finish to write the SparseStateVector (not dynamic), and use it
accordingly in the Synapses code
- More thorough test of set/get


- think about the issue discussed with Romain: hashtable for lookup of values? keep sorted spikes?


** ParameterVector (worst name in the world)

It is only returned by the getattr method of the Synapse object and is
used for the fancy slicing (see doc)

TODO:
- Check the behavior for 3-D slicing

'''
import numpy as np

INITIAL_LEN = 2 # the initial length of the ConstructionSparseStateVector


########################## UTILS ##########################

def dtype2str(dtype):
    '''
    Converts a numpy.dtype object into a string for indexing in
    SparseStateVector's value dictionary
    '''
    return str(dtype)[13:-2]
    
def ensure_iterable(x):
    '''
    Makes sure the value x is iterable, if it's not (a float/int,
    etc...) turn it into an array
    '''
    if not np.iterable(x):
        return np.array([x])
    else:
        return x

def slice2range(sl, n):
    '''
    Given a slice object and a length, returns the array of indices
    concerned by the slice.
    Useful in all the getitem methods!
    '''
    tmp = np.arange(n, dtype = np.int32)
    return ensure_iterable(tmp[sl])

########################## STATE VECTORS ##########################

class SparseStateVector(object):
    '''
    CURRENTLY NOT USED
    
    Compressed (i.e not dynamic) state vectors array
    
    Basically should act as a labeled, static ndarray with possibly different dtypes for different rows.
    '''
    def __init__(self, data, dtypes, row_indices, nvalues, labels = []):
        
        self.dtypes = dtypes
        self.nrows = len(dtypes)
        self.nvalues = nvalues
        
        if labels:
            if not len(labels) == self.nrows:
                raise ValueError('Wrong number of labels')
        
        self.labels = labels
        
        self.row_indices = row_indices
        self._values_dict = data

        
    def __getitem__(self, key):
        ## Before going further I have to know exactly how i'll be using those
        if not isinstance(key, tuple) and len(key) == 2:
            raise IndexError('Wrong index')
        else:
            rows = slice2range(key[0], self.shape[0])
            cols = slice2range(key[1], self.shape[1])
            
            new_dtypes = [typ for i,typ in enumerate(self.dtypes) if i in rows]
            
            for typ in cur_dtypes:
                pass
            
    @property
    def shape(self):
        return (self.nrows, self.nvalues)
            
class ConstructionSparseStateVector(object):
    '''
    This class is used to construct the state vector structure
    underlying the synapses object. It is basically a
    2-D array with possibly different types per row. The rows in this
    object are also labeled, and can be retrieved and set via v.fieldname.
    More importantly, it is dynamic, and doubles its length every time a
    value is entered that reaches the maximum size.
    

    
    ** Initialization ** 
    
    V = ConstructionSparseStateVector(nrows, dtype = int, labels = None)
    
    ``nrows'' is the number of rows (fields) of the data structure
    
    ``dtype'' is either a single, or a tuple of numpy dtypes that
    correspond to the wanted dtypes for each row
    
    ``labels'' when set, one can get each row by accessing the
    object's attribute of the same name
    
    Example:
    ConstructionSparseStateVector(3, dtype = (int32, float32, int32),
                                     labels = ['pre', 'weight', 'post'])
                                     
    
    ** Dynamic Structure **
    
    One can add new values (columns) on the fly to the data structure
    by append them, for example is V is the structure in the above
    example, one can add 2 values as follows:
    
    values = [np.ones(2, dtype = int32), 
              np.randn(2, dtype = float32), 
              np.zeros(2, dtype = int32)]
              
    V.append(values)

    This will add two 1s in the 'pre' field, two random numbers in the
    'weight' field etc...

    ** Accessing the fields **
    
    The recommended way of doing so is to label the field and use its
    label as an attribute. In the above example, we'd use 
    V.pre
    To get the ``pre'' field.
    
    ** To Static ** 
    
    ..automethod:: compress
    
    TODO: Use that!
    
    ** Internals! **
    
    ``_values_dict''
    
    Internally the arrays of the different dtypes are stored into a
    dictionary, and values of the same type in the
    same array.
    
    The function dtype2str converts the dtype object into a string
    (int32 -> 'int32') and the int32 values of the structure are
    accessed internally by self._values_dict['int32']
    
    ``nvalues''
    
    Because of the dynamic structure, the arrays in the dictionary
    can be bigger than there is data in them, hence we keep the number
    of relevant values.
    
    ``row_indices''
    
    row_indices[i] keeps track of the row index of the row i in it's
    dedicated array.
    to get row i we thus do
    
    tmp = self._values_dict[dtype2str(self.dtype[i])]
    return tmp[row_indices[i],:]
    
    I think that's about it!
    '''
    def __init__(self, nrows, dtype = int, labels = None):
        
        self.nrows = nrows
        
        # check dtype:
        if not isinstance(dtype, tuple):
            dtype = (dtype,)*nrows
        if not len(dtype) == nrows:
            raise ValueError('Wrong number of dtypes')
        self.dtypes = dtype
        
        # possibly row labels
        if labels:
            if not len(labels) == nrows:
                raise ValueError('Wrong number of labels')
        self.labels = labels
        self._var_positions = {}

        # take same dtypes and gather them
        nrows_per_typ, self.unique_typ = [], []
        self.row_indices = np.zeros(len(self.dtypes), dtype = int) # will hold the number of the row in the dedicated dtype array
        for i,typ in enumerate(self.dtypes):
            if not typ in self.unique_typ:
                self.unique_typ.append(typ)
                nrows_per_typ.append(0)
            idtyp = self.unique_typ.index(typ)
            self.row_indices[i] = nrows_per_typ[idtyp]

            if labels:
                self._var_positions[labels[i]] = (dtype2str(typ), nrows_per_typ[idtyp])

            nrows_per_typ[idtyp] += 1
            

            
            

        self._values_dict = {}
        # now construct the state nd arrays
        for i,typ in enumerate(self.unique_typ):
            self._values_dict[dtype2str(typ)] = np.zeros((nrows_per_typ[i], INITIAL_LEN), 
                                                          dtype = typ)
            
            
        self.nvalues = 0 # initially no values instantiated
        self.datastruct_size = INITIAL_LEN # datastructure size is the size (nrows) of the arrays
        
    def append(self, values):
        '''
        Values must be a 2-D array containing the pre/post neuron numbers to be added.
        '''
        # where to insert the values
        initial_index = self.nvalues
        final_index = initial_index + values.shape[1]

        # if the new size is bigger that the datastruct size, resize the arrays
        if final_index > self.datastruct_size:
            newsize = 2**np.ceil(np.log2(final_index))
            for typ in self.unique_typ:
                newarray = np.zeros((self._values_dict[dtype2str(typ)].shape[0], 
                                     newsize), dtype = typ)
                newarray[:, :self.nvalues] = self._values_dict[dtype2str(typ)][:, :self.nvalues]
                
                self._values_dict[dtype2str(typ)] = newarray

            self.datastruct_size = newsize

        # appending the pre/post neuron numbers
        pre_typ, pre_row = self._var_positions['_pre']
        post_typ, post_row = self._var_positions['_post']
        self._values_dict[pre_typ][pre_row, initial_index:final_index] = values[0,:]
        self._values_dict[post_typ][post_row, initial_index:final_index] = values[1,:]
        
        self.nvalues = final_index
        
        return initial_index
        
    def compress(self):
        compressed_values_dict = {}
        for i, typ in enumerate(self.dtypes):
            compressed_values_dict[dtype2str(typ)] = self._values_dict[dtype2str(typ)][:, :self.nvalues]
        
        return SparseStateVector(compressed_values_dict, self.dtypes, self.row_indices, self.nvalues, labels = self.labels)
     
    def __getattr__(self, name, default = ''):
        if name == '_allstates':
            return self._values_dict['float32']
        if name in self.labels:
            pre_typ, pre_row = self._var_positions[name] # where to look
            return self._values_dict[pre_typ][pre_row, :self.nvalues]
        try:
            self.__dict__[name]
        except KeyError:
            raise AttributeError('State vector object doesn\'t have a '+name+' attribute')
        
    def __getitem__(self, name):
        if name in self.labels: 
            pre_typ, pre_row = self._var_positions[name] # where to look
            return self._values_dict[pre_typ][pre_row, :self.nvalues]
        else:
            raise KeyError('Item '+name+' not in statevector')
            
        
class ParameterVector(object):
    '''
    Helpful for the setattr of Synapses, 
    
    indeed we want 
    
    synapses.w[slice_pre, slice_post] = 1
    
    to mean
    
    for all the presyn neurons of slice_pre for which there is a synapse, resp, postsyn, set the weight to 1.
    
    hence synapses.w must be of a special class whose __setitem__ method is overriden, otherwise the returned value of synapses.w isn't the right shape etc...
    
    Also we want:

    synapses.w[slice_pre, slice_post, :5] = 2

    to mean

    the 5 first synapses between slice_pre and slice_post neurons = 2
    
    KWD ARGS
    
    ``delay_dt'' is here because in the case of delays, then the value (float with unit) must be converted into timesteps (int), hence to do this this object has to know about the dt.

    '''
    def __init__(self, data, synapses, delay_dt = None):
        self.data = data
        self.groups_shape = (len(synapses.source), len(synapses.target))
        self.synapses = synapses
        self.delay_dt = delay_dt
    
    def __setitem__(self, key, value):
        if isinstance(key, tuple):
            pre_synapses = self.synapses.pre2synapses(slice2range(key[0] , self.groups_shape[0]))
            post_synapses = self.synapses.post2synapses(slice2range(key[0] , self.groups_shape[0]))
                                                      
            indices = np.intersect1d(pre_synapses, post_synapses)

            # Shite! now I do I do that??!
            if len(key) > 2:
                indices = indices[slice2range(key[2], len(indices))] 

            if self.delay_dt:
                value = np.array(np.round(value/self.delay_dt), dtype = self.data.dtype)

            self.data[indices] = value
        else:
            raise IndexError
        

    def __getitem__(self, key):
        return self.data.__getitem__(key)

    def __repr__(self):
        return self.data.__repr__()

    def __str__(self):
        return self.data.__repr__()
