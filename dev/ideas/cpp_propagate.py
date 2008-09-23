from numpy import *
from scipy import weave

l = [array([1.,2.]), array([3.,4.])]
#l = [1.,2.]
m = array([0.,0.])

print m

code = """
       //m(0) = (double)l[0];
       PyArrayObject* l0 = convert_to_numpy(l[0],"l0");
       //conversion_numpy_check_type(l0,PyArray_DOUBLE,"l0");
       //conversion_numpy_check_size(l0,1,"l0");
       blitz::Array<double,1> l0b = convert_to_blitz<double,1>(l0,"l0");
       m(0)=l0b(1);
       m(1)=(double)l0b.numElements();
       """

#        py_m = get_variable("m",raw_locals,raw_globals);
#        PyArrayObject* m_array = convert_to_numpy(py_m,"m");
#        conversion_numpy_check_type(m_array,PyArray_DOUBLE,"m");
#        conversion_numpy_check_size(m_array,1,"m");
#        blitz::Array<double,1> m = convert_to_blitz<double,1>(m_array,"m");
#        blitz::TinyVector<int,1> Nm = m.shape();

weave.inline(code,['l','m'],
             compiler='gcc',
             type_converters=weave.converters.blitz,
             verbose=2)

print m

#            code =  """
#                    int numspikes=0;
#                    for(int i=0;i<N;i++)
#                        if(V(i)>Vt)
#                            spikes(numspikes++) = i;
#                    return_val = numspikes;
#                    """
#            # WEAVE NOTE: set the environment variable USER if your username has a space
#            # in it, say set USER=DanGoodman if your username is Dan Goodman, this is
#            # because weave uses this to create file names, but doesn't correctly send these
#            # values to the compiler, causing problems.
#            numspikes = weave.inline(code,['spikes','V','Vt','N'],\
#                                     compiler=self._cpp_compiler,
#                                     type_converters=weave.converters.blitz)
