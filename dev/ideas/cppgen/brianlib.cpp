#include "brianlib.h" 
#include<sstream>
#include<iostream>
using namespace std;

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
		this->last_spikes = this->thr->__call__(this);
	if(this->reset)
		this->reset->__call__(this);
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
    //for(int i=0;i<m;i++) cout << c[i] << endl;
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
			out << A[i+j*m] << " ";
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
	for(int i=0;i<group->S_m;i++)
		if(group->S[this->state+group->S_n*i]>this->value)
			spikelist.push_back(i);
	return spikelist;
}

void Reset::__call__(NeuronGroup *group)
{
	SpikeList &spikelist = group->last_spikes;
	for(SpikeList::iterator i=spikelist.begin();i!=spikelist.end();i++)
		group->S[this->state+group->S_n*(*i)] = this->value;
}

StateMonitor::StateMonitor(NeuronGroup *group, int state)
{
	this->group = group;
	this->state = state;
	for(int i=0;i<group->S_m;i++)
		this->values.push_back(vector<double>());
}

void StateMonitor::__call__()
{
	for(int i=0;i<group->S_m;i++)
		this->values[i].push_back(this->group->S[this->state+this->group->S_n*i]);
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
	for(list<NeuronGroup*>::iterator i=this->groups.begin();i!=this->groups.end();i++)
		(*i)->update();
	for(list<NetworkOperation*>::iterator i=this->operations.begin();i!=this->operations.end();i++)
		(*i)->__call__();
}

void Network::run(int timesteps)
{
	for(int i=0;i<timesteps;i++)
		this->update();
}