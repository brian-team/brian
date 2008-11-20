%module brianlib

%include "std_string.i"
%include "std_vector.i"
%include "std_list.i"

%{
#define SWIG_FILE_WITH_INIT
#include "brianlib.h"
%}

namespace std
{
    %template(SpikeList) list<int>;
    %template(VectorDouble) vector<double>;
}

%include "numpy.i"
%init %{
import_array();
%}

%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {(double *S, int n, int m)};
%apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {(double *M, int M_n, int M_m)};
%apply (double* INPLACE_ARRAY1, int DIM1) {(double *b, int b_n)};
%apply (double* ARGOUT_ARRAY1, int DIM1) {(double *S_out_flat, int nm)};

%include "brianlib.h"
