#include "brianlib.h"

/*
void Reset::__call__(NeuronGroup *group)
{
	//SpikeList &spikelist = group->last_spikes;
	SpikeList spikelist = group->LS->lastspikes();
	for(SpikeList::iterator i=spikelist.begin();i!=spikelist.end();i++)
		//group->S[this->state+group->num_vars*(*i)] = this->value;
		neuron_value(group, *i, this->state) = this->value;
//	V[P.LS.lastspikes()] = self.resetvalue
}
*/

void Reset::__call__(NeuronGroup *group)
{
	int *ret;
	int ret_n;
	group->LS->lastspikes(&ret, &ret_n);
	for(int i=0;i<ret_n;i++)
		neuron_value(group, i, this->state) = this->value;
}

/*
void Refractoriness::__call__(NeuronGroup *group)
{
	SpikeList spikelist = group->LS->__getslice__(0, period);
	for(SpikeList::iterator i=spikelist.begin();i!=spikelist.end();i++)
		neuron_value(group, *i, this->state) = this->value;
//  V[P.LS[0:period]] = self.resetvalue
}
*/

void Refractoriness::__call__(NeuronGroup *group)
{
	int *ret;
	int ret_n;
	group->LS->__getslice__(&ret, &ret_n, 0, this->period);
	for(int i=0;i<ret_n;i++)
		neuron_value(group, ret[i], this->state) = this->value;
}