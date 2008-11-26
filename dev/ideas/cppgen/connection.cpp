#include "brianlib.h"

void ConnectionMatrix::add_rows(int *x, int n, double *b, int b_n)
{
	for(int i=0;i<n;i++)
		this->add_row(x[i], b, b_n);
}

void DenseConnectionMatrix::add_row(int i, double *b, int b_n)
{
	for(int j=0;j<b_n;j++)
		b[j] += matrix_value(this->W, i, j, this->n, this->m);
}

void Connection::do_propagate()
{
	int *x, n;
	this->source->get_spikes(&x, &n, this->delay);
	this->propagate(x, n);
}

void Connection::propagate(int *x, int n)
{
	this->connmat->add_rows(x, n,
				&(neuron_value(this->target, this->target->origin, this->state)),
				this->target->num_neurons);
}
