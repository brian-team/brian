%module cspikequeue
%include "exception.i"
%include "std_string.i"

%exception {
	try {
		$action
	} catch( std::runtime_error &e ) {
		PyErr_SetString(PyExc_RuntimeError, const_cast<char *>(e.what()));
		return NULL;
	}
}

%allowexception;


%{
    #define SWIG_FILE_WITH_INIT
    #include "spikequeue.h"
%}

%include "numpy.i"

%init %{
    import_array();
%}



%apply (int DIM1, long* IN_ARRAY1) {(int len1, long* vec1), (int len2, long* vec2)}

%apply (long** ARGOUTVIEW_ARRAY1, int* DIM1 ) {(long **ret, int *ret_n)};

%include "spikequeue.h"
