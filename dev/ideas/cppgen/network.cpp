#include "brianlib.h"

void Network::add(NeuronGroup *group)
{
	this->groups.push_back(group);
}

void Network::add(NetworkOperation *op)
{
	this->operations.push_back(op);
}

void Network::add(Connection *conn)
{
	this->connections.push_back(conn);
}

void Network::update()
{
	// Groups
	for(list<NeuronGroup*>::iterator i=this->groups.begin();i!=this->groups.end();i++)
		(*i)->update();
	// Connections
	for(list<Connection*>::iterator i=this->connections.begin();i!=this->connections.end();i++)
		(*i)->do_propagate();
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

