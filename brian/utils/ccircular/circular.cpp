#include "ccircular.h"
#include<string.h>
#include<sstream>

CircularVector::CircularVector(int n)
{
	this->X = NULL;
	this->retarray = NULL;
	this->n = n;
	this->X = new long[n]; // we don't worry about memory errors for the moment...
	this->retarray = new long[n];
	if(!this->X || !this->retarray){
		if(this->X) {
			delete [] this->X;
			this->X = 0;
		}
		if(this->retarray) {
			delete [] this->retarray;
			this->retarray = 0;
		}
		throw BrianException("Not enough memory in creating CircularVector.");
	}
	this->reinit();
}

CircularVector::~CircularVector()
{
	if(this->X) delete [] this->X;
	if(this->retarray) delete [] this->retarray;
	this->X = NULL;
	this->retarray = NULL;
}

void CircularVector::expand(long n)
{
	long orig_n = this->n;
	this->n += n;
	n = this->n;
	long *new_X = new long[n];
	long *new_retarray = new long[n];
	if(!new_X || !new_retarray){
		if(new_X) delete [] new_X;
		if(new_retarray) delete [] new_retarray;
		throw BrianException("Not enough memory in expanding CircularVector.");
	}
	// newS.X[:S.n-S.cursor] = S.X[S.cursor:]
	memcpy((void *)new_X, (void *)(this->X+this->cursor), sizeof(long)*(orig_n-this->cursor));
	// newS.X[S.n-S.cursor:S.n] = S.X[:S.cursor]
	memcpy((void *)(new_X+orig_n-this->cursor), (void *)(this->X), sizeof(long)*this->cursor);
	this->cursor = orig_n;
	delete [] this->X;
	this->X = new_X;
	delete [] this->retarray;
	this->retarray = new_retarray;
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

void CircularVector::__getslice__(long **ret, int *ret_n, int i, int j)
{
	int i0 = this->index(i);
	int j0 = this->index(j);
	int n = 0;
	for(int k=i0;k!=j0;k=(k+1)%this->n)
		this->retarray[n++] = this->X[k];
	*ret = this->retarray;
	*ret_n = n;
}

void CircularVector::get_conditional(long **ret, int *ret_n, int i, int j, int min, int max, int offset)
{
	int i0, j0;
	int lo, mid, hi;
	int ioff = this->index(i);
	int joff = this->index(j);
	int lensearch;
	if(joff>=ioff)
		lensearch = joff-ioff;
	else
		lensearch = this->n-(ioff-joff);
	// start with a binary search to the left, note that we use here the bisect_left
	// algorithm from the standard python library translated into C++ code, and
	// altered to work with a circular array with an offset. The lo and hi variables
	// vary from 0 to lensearch but the probe this->X[...] is offset by ioff, the
	// position in the X array of the i variable passed to the function.
	lo = 0;
	hi = lensearch;
    while(lo<hi)
    {
        mid = (lo+hi)/2;
        if(this->X[(mid+ioff)%this->n]<min)
        	lo = mid+1;
        else
        	hi = mid;
    }
    i0 = (lo+ioff)%this->n;
	// then a binary search to the right
    //lo = 0; // we can use the lo output from the previous step
    hi = lensearch;
    while(lo<hi)
    {
        mid = (lo+hi)/2;
        if(this->X[(mid+ioff)%this->n]<max)
        	lo = mid+1;
        else
        	hi = mid;
    }
    j0 = (lo+ioff)%this->n;
	// then fill in the return array
	int n = 0;
	for(int k=i0;k!=j0;k=(k+1)%this->n)
		this->retarray[n++] = this->X[k]-offset;
	*ret = this->retarray;
	*ret_n = n;
}

void CircularVector::__setslice__(int i, int j, long *x, int n)
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

SpikeContainer::SpikeContainer(int m)
{
	try{
		this->S = NULL;
		this->ind = NULL;
		this->S = new CircularVector(2);
		this->remaining_space = 1;
		if(m<2) m=2;
		this->ind = new CircularVector(m+1);
	} catch(BrianException &e) {
		if(this->S){
			delete this->S;
			this->S = 0;
		}
		if(this->ind){
			delete this->ind;
			this->ind = 0;
		}
		throw;
	}
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

void SpikeContainer::push(long *y, int n)
{
	long freed_space = (this->ind->__getitem__(2)-this->ind->__getitem__(1))%this->S->n;
	if(freed_space<0) freed_space += this->S->n;
	this->remaining_space += freed_space;
	while(n>=this->remaining_space){
		long orig_cursor = this->S->cursor;
		long orig_n = this->S->n;
		this->S->expand(this->S->n); // double size of S
		for(long i=0; i<this->ind->n; i++){
			this->ind->X[i] = (this->ind->X[i]-orig_cursor)%orig_n;
			if(this->ind->X[i]<0) this->ind->X[i] += orig_n;
			if(this->ind->X[i]==0) this->ind->X[i] = orig_n;
		}
		this->remaining_space += orig_n;
	}
	this->S->__setslice__(0, n, y, n);
	this->S->advance(n);
	this->ind->advance(1);
	this->ind->__setitem__(0, this->S->cursor);
	this->remaining_space -= n;
}

void SpikeContainer::lastspikes(long **ret, int *ret_n)
{
	this->S->__getslice__(ret, ret_n, this->ind->__getitem__(-1)-this->S->cursor, this->S->n);
}

void SpikeContainer::__getitem__(long **ret, int *ret_n, int i)
{
	this->S->__getslice__(ret, ret_n, this->ind->__getitem__(-i-1)-this->S->cursor,
								 this->ind->__getitem__(-i)-this->S->cursor+this->S->n);
}

void SpikeContainer::get_spikes(long **ret, int *ret_n, int delay, int origin, int N)
{
	return this->S->get_conditional(ret, ret_n,
			this->ind->__getitem__(-delay-1)-this->S->cursor,
			this->ind->__getitem__(-delay)-this->S->cursor+this->S->n,
			origin, origin+N, origin);
}

void SpikeContainer::__getslice__(long **ret, int *ret_n, int i, int j)
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

