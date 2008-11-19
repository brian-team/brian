%module brianlib

%include "std_string.i"
%include "std_vector.i"

%{
#include "brianlib.h"
%}

namespace std {
   %template(DoubleVector) vector<double>;
}

%include "brianlib.h"
