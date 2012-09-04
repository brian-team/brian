%module cspikequeue
%include "exception.i"

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



%apply (long* ARGOUT_ARRAY1, int DIM1 ) {(long *ret_out, int ret_n_out)};
%apply (long** ARGOUTVIEW_ARRAY1, int* DIM1 ) {(long **ret, int *ret_n)};

%include "spikequeue.h"
