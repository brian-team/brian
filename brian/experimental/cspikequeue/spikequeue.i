%module cspikequeue

%{
    #define SWIG_FILE_WITH_INIT
    #include "spikequeue.h"
%}

%include "numpy.i"

%init %{
    import_array();
%}



%apply (long** ARGOUTVIEW_ARRAY1, int* DIM1 ) {(long **ret, int *ret_n)};
%apply (long* ARGOUT_ARRAY1, int DIM1 ) {(long *ret_out, int ret_n_out)};

%include "spikequeue.h"
