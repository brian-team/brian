#include "fexp.h"

void fastexp(double *x, int n, double *y, int m)
{
	for(;n;n--) *y++ = fexp(*x++);
}
