#include "brianlib.h" 
#include<sstream>
using namespace std;

void NeuronGroup::get_S_flat(double *S_out_flat, int nm)
{
	for(int i=0;i<nm;i++)
		S_out_flat[i] = this->S[i];
}

void LinearStateUpdater::__call__(NeuronGroup *group)
{
//    n = len(P)
//    m = len(self)
//    S = P._S
//    A = self.A
//    c = self._C
	int m = this->M_n;
	int n = group->S_m;
	double *S = group->S;
	double *A = this->M;
	double *c = this->b;
    double x[m];
    for(int i=0;i<n;i++)  
    {
        for(int j=0;j<m;j++)
        {
            x[j] = c[j];
            for(int k=0;k<m;k++)
                //x[j] += A(j,k) * S(k,i);
            	x[j] += A[j+k*m] * S[k+i*m];
        }
        for(int j=0;j<m;j++)
            S[j+i*m] = x[j];
    }	
}
