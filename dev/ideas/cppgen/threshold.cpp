#include "brianlib.h"
/*
SpikeList Threshold::__call__(NeuronGroup *group)
{
	SpikeList spikelist;
	for(int i=0;i<group->num_neurons;i++)
		//if(group->S[this->state+group->num_vars*i]>this->value)
		if(neuron_value(group, i, this->state)>this->value)
			spikelist.push_back(i);
	return spikelist;
}
*/
void Threshold::__call__(int **ret, int *ret_n, NeuronGroup *group)
{
	int n = 0;
	for(int i=0;i<group->num_neurons;i++)
		if(neuron_value(group, i, this->state)>this->value)
			group->spikesarray[n++] = i;
	*ret = group->spikesarray;
	*ret_n = n; 
}
