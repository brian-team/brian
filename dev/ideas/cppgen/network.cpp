#include "brianlib.h"

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

