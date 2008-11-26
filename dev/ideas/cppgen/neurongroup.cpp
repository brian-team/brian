#include "brianlib.h"

NeuronGroup::NeuronGroup(double *S, int n, int m, StateUpdater *su,
						 Threshold *thr, ResetBase *reset,
						 int ls_n, int ls_m)
{
	this->S = S;
	this->num_vars = n;
	this->num_neurons = m;
	this->S_m = m;
	this->su = su;
	this->thr = thr;
	this->resetobj = reset;
	this->LS = new SpikeContainer(ls_n, ls_m);
	this->spikesarray = new int [this->num_neurons];
	
	this->owner = this;
	this->origin = 0;
}

NeuronGroup::NeuronGroup(NeuronGroup *parent, int origin, int length)
{
	this->owner = parent;
	this->origin = origin;
	
	this->S = parent->S;
	this->num_vars = parent->num_vars;
	this->num_neurons = length;
	this->S_m = parent->S_m;
	this->su = NULL;
	this->thr = NULL;
	this->resetobj = NULL;
	this->LS = parent->LS;
	this->spikesarray = parent->spikesarray;
}

NeuronGroup::~NeuronGroup()
{
	if(this->owner==this)
	{
		if(this->LS) delete this->LS;
		if(this->spikesarray) delete [] this->spikesarray;
		this->LS = NULL;
		this->spikesarray = NULL;
	}
}
/*
void NeuronGroup::get_S_flat(double *S_out_flat, int nm)
{
	for(int i=0;i<nm;i++)
		S_out_flat[i] = this->S[i];
}
*/
void NeuronGroup::update()
{
	if(this->owner==this)
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
}

void NeuronGroup::get_spikes(int **ret, int *ret_n, int delay)
{
	this->LS->get_spikes(ret, ret_n, delay, this->origin, this->num_neurons);
}

void NeuronGroup::reset()
{
	if(this->resetobj)
		this->resetobj->__call__(this);
}
