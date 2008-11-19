#ifndef _BRIAN_LIB_H
#define _BRIAN_LIB_H

#include<vector>
#include<string>
using namespace std;

class NeuronGroup
{
public:
	double *S;
	int S_n, S_m;
	NeuronGroup(double *S, int n, int m) : S(S), S_n(n), S_m(m) {}
	void get_S_flat(double *S_out_flat, int nm);
};

class LinearStateUpdater
{
public:
	double *M;
	int M_n, M_m;
	double *b;
	int b_n;
	LinearStateUpdater(double *M, int M_n, int M_m, double *b, int b_n) :
		M(M), M_n(M_n), M_m(M_m), b(b), b_n(b_n) {}
	void __call__(NeuronGroup *group); // like in Python
};

#endif 
