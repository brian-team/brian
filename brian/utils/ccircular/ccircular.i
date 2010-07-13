%module ccircular
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
#include "ccircular.h"
%}

%include "numpy.i"
%init %{
	import_array();
%}

%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {(double *S, int n, int m)};
%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {(double *M, int M_n, int M_m)};
%apply (long* INPLACE_ARRAY1, int DIM1) {(long *x, int n)};
%apply (long* INPLACE_ARRAY1, int DIM1) {(long *y, int n)};
%apply (double* INPLACE_ARRAY1, int DIM1) {(double *b, int b_n)};
%apply (double* ARGOUT_ARRAY1, int DIM1) {(double *S_out_flat, int nm)};
%apply (long** ARGOUTVIEW_ARRAY1, int* DIM1 ) {(long **ret, int *ret_n)};

%include "ccircular.h"
