#include "brianlib.h"
#include<sstream>

void LinearStateUpdater::__call__(NeuronGroup *group)
{
//    n = len(P)
//    m = len(self)
//    S = P._S
//    A = self.A
//    c = self._C
	int m = this->M_n;
	int n = group->num_neurons;
	double *A = this->M;
	double *c = this->b;
    double x[m];
    //for(int i=0;i<m;i++) cout << c[i] << endl;
    for(int i=0;i<n;i++)  
    {
        for(int j=0;j<m;j++)
        {
            x[j] = c[j];
            for(int k=0;k<m;k++)
                //x[j] += A(j,k) * S(k,i);
            	//x[j] += A[j+k*m] * S[k+i*m];
            	//x[j] += A[k+j*m] * S[i+k*n];
            	x[j] += A[k+j*m] * neuron_value(group, i, k);
        }
        for(int j=0;j<m;j++)
            //S[j+i*m] = x[j];
        	//S[i+j*n] = x[j];
        	neuron_value(group, i, j) = x[j];
    }	
}

string LinearStateUpdater::__repr__()
{
	int m = this->M_n;
	double *A = this->M;
	double *c = this->b;
	stringstream out;
	out << "LinearStateUpdater" << endl;
	out << "A = " << endl;
	for(int i=0;i<m;i++)
	{
		for(int j=0;j<m;j++)
			out << A[j+i*m] << " ";
		out << endl;
	}
	out << "c = ";
	for(int i=0;i<m;i++)
		out << c[i] << " ";
	return out.str();	
}
