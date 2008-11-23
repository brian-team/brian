#include "brianlib.h" 
#include<sstream>
#include<iostream>
using namespace std;

#define neuron_value(group, neuron, state) group->S[neuron+state*(group->num_neurons)]

NeuronGroup::NeuronGroup(double *S, int n, int m, StateUpdater *su,
						 Threshold *thr, ResetBase *reset,
						 int ls_n, int ls_m)
{
	this->S = S;
	this->num_vars = n;
	this->num_neurons = m;
	this->su = su;
	this->thr = thr;
	this->resetobj = reset;
	this->LS = new SpikeContainer(ls_n, ls_m);
}
/*
inline double& neuron_value(NeuronGroup *group, int neuron, int state)
{
	return group->S[neuron+state*(group->num_neurons)];
}
*/
NeuronGroup::~NeuronGroup()
{
	if(this->LS) delete this->LS;
}

void NeuronGroup::get_S_flat(double *S_out_flat, int nm)
{
	for(int i=0;i<nm;i++)
		S_out_flat[i] = this->S[i];
}

void NeuronGroup::update()
{
	if(this->su)
		this->su->__call__(this);
	if(this->thr)
	{
		SpikeList spikes = this->thr->__call__(this);
		if(this->LS)
			this->LS->push(spikes);
	}
}

SpikeList NeuronGroup::get_spikes(int delay)
{
	// TODO: this assumes that this is not a subgroup
	return this->LS->get_spikes(delay, 0, this->num_neurons);
}

void NeuronGroup::reset()
{
	if(this->resetobj)
		this->resetobj->__call__(this);
}

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

SpikeList Threshold::__call__(NeuronGroup *group)
{
	SpikeList spikelist;
	for(int i=0;i<group->num_neurons;i++)
		//if(group->S[this->state+group->num_vars*i]>this->value)
		if(neuron_value(group, i, this->state)>this->value)
			spikelist.push_back(i);
	return spikelist;
}

void Reset::__call__(NeuronGroup *group)
{
	//SpikeList &spikelist = group->last_spikes;
	SpikeList spikelist = group->LS->lastspikes();
	for(SpikeList::iterator i=spikelist.begin();i!=spikelist.end();i++)
		//group->S[this->state+group->num_vars*(*i)] = this->value;
		neuron_value(group, *i, this->state) = this->value;
//	V[P.LS.lastspikes()] = self.resetvalue
}

void Refractoriness::__call__(NeuronGroup *group)
{
	SpikeList spikelist = group->LS->__getslice__(0, period);
	for(SpikeList::iterator i=spikelist.begin();i!=spikelist.end();i++)
		neuron_value(group, *i, this->state) = this->value;
//  V[P.LS[0:period]] = self.resetvalue
}

StateMonitor::StateMonitor(NeuronGroup *group, int state)
{
	this->group = group;
	this->state = state;
	for(int i=0;i<group->num_neurons;i++)
		this->values.push_back(vector<double>());
}

void StateMonitor::__call__()
{
	for(int i=0;i<group->num_neurons;i++)
		//this->values[i].push_back(this->group->S[this->state+this->group->num_vars*i]);
		//this->values[i].push_back(this->group->S[i+this->state*this->group->num_neurons]);
		this->values[i].push_back(neuron_value(this->group, i, this->state));
}

vector<double> StateMonitor::__getitem__(int i)
{
	return this->values[i];
}

void Network::add(NeuronGroup *group)
{
	this->groups.push_back(group);
}

void Network::add(NetworkOperation *op)
{
	this->operations.push_back(op);
}

void Network::update()
{
	// Groups
	for(list<NeuronGroup*>::iterator i=this->groups.begin();i!=this->groups.end();i++)
		(*i)->update();
	// TODO: Connections
	// Resets
	for(list<NeuronGroup*>::iterator i=this->groups.begin();i!=this->groups.end();i++)
		(*i)->reset();
	// End operations
	for(list<NetworkOperation*>::iterator i=this->operations.begin();i!=this->operations.end();i++)
		(*i)->__call__();
}

void Network::run(int timesteps)
{
	for(int i=0;i<timesteps;i++)
		this->update();
}

CircularVector::CircularVector(int n)
{
	this->n = n;
	this->X = new int[n]; // we don't worry about memory errors for the moment...
	this->reinit();
}

CircularVector::~CircularVector()
{
	if(this->X) delete [] this->X;
	this->X = NULL;
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

list<int> CircularVector::__getslice__(int i, int j)
{
	int i0 = this->index(i);
	int j0 = this->index(j);
	list<int> slice;
	for(int k=i0;k!=j0;k=(k+1)%this->n)
		slice.push_back(this->X[k]);
	return slice;
}

// This can potentially be sped up substantially using a bisection algorithm
list<int> CircularVector::get_conditional(int i, int j, int min, int max, int offset)
{
	int i0 = this->index(i);
	int j0 = this->index(j);
	list<int> slice;
	for(int k=i0;k!=j0;k=(k+1)%this->n)
	{
		int Xk = this->X[k];
		if(Xk>=min && Xk<max)
			slice.push_back(Xk-offset);
	}
	return slice;
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

void CircularVector::__setslice__(int i, int j, list<int> &x)
{
	if(j>i)
	{
		int i0 = this->index(i);
		int j0 = this->index(j);
		list<int>::iterator l = x.begin();
		for(int k=i0;k!=j0;k=(k+1)%this->n,l++)
			this->X[k] = *l;
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

void SpikeContainer::push(list<int> &x)
{
	int n = x.size();
	this->S->__setslice__(0, n, x);
	this->S->advance(n);
	this->ind->advance(1);
	this->ind->__setitem__(0, this->S->cursor);
}

SpikeList SpikeContainer::lastspikes()
{
	return this->S->__getslice__(this->ind->__getitem__(-1)-this->S->cursor, this->S->n);
}

SpikeList SpikeContainer::__getitem__(int i)
{
	return this->S->__getslice__(this->ind->__getitem__(-i-1)-this->S->cursor,
								 this->ind->__getitem__(-i)-this->S->cursor+this->S->n);
}

SpikeList SpikeContainer::get_spikes(int delay, int origin, int N)
{
	return this->S->get_conditional(
			this->ind->__getitem__(-delay-1)-this->S->cursor,
			this->ind->__getitem__(-delay)-this->S->cursor+this->S->n,
			origin, origin+N, origin);
}

SpikeList SpikeContainer::__getslice__(int i, int j)
{
	return this->S->__getslice__(
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

