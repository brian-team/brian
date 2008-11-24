#include "brianlib.h"

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
	this->spikesarray = new int [this->num_neurons];
}

NeuronGroup::~NeuronGroup()
{
	if(this->LS) delete this->LS;
	if(this->spikesarray) delete [] this->spikesarray;
	this->LS = NULL;
	this->spikesarray = NULL;
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
		int *ret;
		int ret_n;
		//SpikeList spikes = this->thr->__call__(this);
		this->thr->__call__(&ret, &ret_n, this);
		if(this->LS)
			//this->LS->push(spikes);
			this->LS->push(ret, ret_n);
	}
}

/*
SpikeList NeuronGroup::get_spikes(int delay)
{
	// TODO: this assumes that this is not a subgroup
	return this->LS->get_spikes(delay, 0, this->num_neurons);
}
*/
void NeuronGroup::get_spikes(int **ret, int *ret_n, int delay)
{
	this->LS->get_spikes(ret, ret_n, delay, 0, this->num_neurons);
}

void NeuronGroup::reset()
{
	if(this->resetobj)
		this->resetobj->__call__(this);
}