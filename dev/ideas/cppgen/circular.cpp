#include "brianlib.h"
#include<sstream>

CircularVector::CircularVector(int n)
{
	this->n = n;
	this->X = new int[n]; // we don't worry about memory errors for the moment...
	this->retarray = new int[n];
	this->reinit();
}

CircularVector::~CircularVector()
{
	if(this->X) delete [] this->X;
	if(this->retarray) delete [] this->retarray;
	this->X = NULL;
	this->retarray = NULL;
}

void CircularVector::reinit()
{
	this->cursor = 0;
	for(int i=0;i<this->n;i++)
		this->X[i] = 0;
}

void CircularVector::advance(int k)
{
	this->cursor = this->index(k);
}

int CircularVector::__len__()
{
	return this->n;
}

inline int CircularVector::index(int i)
{
	int j = (this->cursor+i)%this->n;
	if(j<0) j+=this->n;
	return j;
}
inline int CircularVector::getitem(int i)
{
	return this->X[this->index(i)];
}
int CircularVector::__getitem__(int i)
{
	return this->getitem(i);
}

void CircularVector::__setitem__(int i, int x)
{
	this->X[this->index(i)] = x;
}

void CircularVector::__getslice__(int **ret, int *ret_n, int i, int j)
{
	int i0 = this->index(i);
	int j0 = this->index(j);
	int n = 0;
	for(int k=i0;k!=j0;k=(k+1)%this->n)
		this->retarray[n++] = this->X[k];
	*ret = this->retarray;
	*ret_n = n;
}

// This can potentially be sped up substantially using a bisection algorithm
void CircularVector::get_conditional(int **ret, int *ret_n, int i, int j, int min, int max, int offset)
{
	int i0 = this->index(i);
	int j0 = this->index(j);
	int n = 0;
	for(int k=i0;k!=j0;k=(k+1)%this->n)
	{
		int Xk = this->X[k];
		if(Xk>=min && Xk<max)
			this->retarray[n++] = Xk-offset;
	}
	*ret = this->retarray;
	*ret_n = n;
}

void CircularVector::__setslice__(int i, int j, int *x, int n)
{
	if(j>i)
	{
		int i0 = this->index(i);
		int j0 = this->index(j);
		for(int k=i0,l=0;k!=j0 && l<n;k=(k+1)%this->n,l++)
			this->X[k] = x[l];
	}
}

string CircularVector::__repr__()
{
	stringstream out;
	out << "CircularVector(";
	out << "cursor=" << this->cursor;
	out << ", X=[";
	for(int i=0;i<this->n;i++)
	{
		if(i) out << " ";
		out << this->X[i];
	}
	out << "])";
	return out.str();
}
string CircularVector::__str__()
{
	return this->__repr__();
}

SpikeContainer::SpikeContainer(int n, int m)
{
	this->S = new CircularVector(n+1);
	this->ind = new CircularVector(m+1);
}

SpikeContainer::~SpikeContainer()
{
	if(this->S) delete this->S;
	if(this->ind) delete this->ind;
}

void SpikeContainer::reinit()
{
	this->S->reinit();
	this->ind->reinit();
}

void SpikeContainer::push(int *y, int n)
{
	this->S->__setslice__(0, n, y, n);
	this->S->advance(n);
	this->ind->advance(1);
	this->ind->__setitem__(0, this->S->cursor);
}

void SpikeContainer::lastspikes(int **ret, int *ret_n)
{
	this->S->__getslice__(ret, ret_n, this->ind->__getitem__(-1)-this->S->cursor, this->S->n);
}

void SpikeContainer::__getitem__(int **ret, int *ret_n, int i)
{
	this->S->__getslice__(ret, ret_n, this->ind->__getitem__(-i-1)-this->S->cursor,
								 this->ind->__getitem__(-i)-this->S->cursor+this->S->n);
}

void SpikeContainer::get_spikes(int **ret, int *ret_n, int delay, int origin, int N)
{
	return this->S->get_conditional(ret, ret_n,
			this->ind->__getitem__(-delay-1)-this->S->cursor,
			this->ind->__getitem__(-delay)-this->S->cursor+this->S->n,
			origin, origin+N, origin);
}

void SpikeContainer::__getslice__(int **ret, int *ret_n, int i, int j)
{
	return this->S->__getslice__(ret, ret_n,
			this->ind->__getitem__(-j)-this->S->cursor,
			this->ind->__getitem__(-i)-this->S->cursor+this->S->n);	
}

string SpikeContainer::__repr__()
{
	stringstream out;
	out << "SpikeContainer(" << endl;
	out << "  S: ";
	out << this->S->__repr__() << endl;
	out << "  ind: ";
	out << this->ind->__repr__(); 
	out << ")";
	return out.str();
}
string SpikeContainer::__str__()
{
	return this->__repr__();
}

