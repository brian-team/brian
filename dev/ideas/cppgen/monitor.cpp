#include "brianlib.h"

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
		this->values[i].push_back(neuron_value(this->group, i, this->state));
}

vector<double> StateMonitor::__getitem__(int i)
{
	return this->values[i];
}

