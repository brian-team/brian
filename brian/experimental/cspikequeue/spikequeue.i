%module cspikequeue

%{
    #define SWIG_FILE_WITH_INIT
    #include "spikequeue.h"
%}

%include "numpy.i"

%init %{
    import_array();
%}



%apply (long *ARGOUTVIEW_ARRAY1, int DIM1 ) {(long **ret, int *ret_n)};

%include "spikequeue.h"
