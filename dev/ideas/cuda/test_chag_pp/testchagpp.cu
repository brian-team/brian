#include "testchagpp.h"
#include <chag/pp/compact.cuh>
namespace pp = chag::pp;

struct Predicate
{
	__device__ bool operator() (double value) const
	{
		return value>0.0;
	}
};

void find_positive(
		int x_gpu_start,
		int x_gpu_end,
		int y_gpu_start,
		int count_start
		)
{
	pp::compact(
	    (double *)x_gpu_start,              /* Input start pointer */
	    (double *)x_gpu_end,     /* Input end pointer */
	    (double *)y_gpu_start,              /* Output start pointer */
	    (size_t *)count_start,            /* Storage for valid element count */
	    Predicate()             /* Predicate */
	    );
}
