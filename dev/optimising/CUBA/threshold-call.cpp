/*
 * Calling with netsize = 4000 and repeats = 100000 takes about 11s compared to
 * Python's 92s of which NumPy's 40s so this looks to be a candidate for optimisation
 * 
 */

#include<cstdlib>
#include<ctime>
#include<iostream>
#include<list>

const int netsize = 4000; // 4000 for comparison with cuba-profile.txt
const int repeats = 50000; // 400000 for comparison with cuba-profile.txt

using namespace std;

double timeforfiringfraction(double *V, double Vt)
{
	clock_t start = clock();
	list<int> l;
	for(int i=0;i<repeats;i++)
	{
		l.clear();
		for(int j=0;j<netsize;j++)
		{
			if(V[j]>Vt) l.push_back(j);
		}
	}
	return (clock()-start)*1./CLOCKS_PER_SEC;
}

// This is the method used by the .nonzero() method of NumPy
double timeforfiringfraction2(double *V, double Vt)
{
	clock_t start = clock();
	for(int i=0;i<repeats;i++)
	{
		int count = 0;
		for(int j=0;j<netsize;j++)
			if(V[j]>Vt) count++;
		int *l = new int [count];
		int *lptr = l;
		for(int j=0;j<netsize;j++)
		{
			if(V[j]>Vt) *lptr++ = j;
		}
		delete l;
	}
	return (clock()-start)*1./CLOCKS_PER_SEC;
}

int main(void)
{
	double V[netsize];
	srand((unsigned int)time(NULL));
	for(int i=0;i<netsize;i++)
		V[i]=double(rand())/(double)RAND_MAX;
	int netrateinc = 1;
	for(int netrate=1;netrate<=400;netrate+=netrateinc)
	{
		if(10<=netrate & netrate<100)
			netrateinc = 10;
		if(100<=netrate)
			netrateinc = 50;
		double Vt = 1 - (double)netrate/4000.;
		cout << netrate << ",  " << timeforfiringfraction(V,Vt) << ", " << timeforfiringfraction2(V,Vt) << endl;
	}
	return 0;
} 
