%module templatecontainer

%{
#define SWIG_FILE_WITH_INIT
#include "templatecontainer.h"
%}

%include "templatecontainer.h"

%template(TemplateContainerInt) TemplateContainer<int>;
%template(TemplateContainerDouble) TemplateContainer<double>;
