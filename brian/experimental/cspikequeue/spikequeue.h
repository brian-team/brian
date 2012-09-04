#ifndef _BRIAN_LIB_H
#define _BRIAN_LIB_H

#include<vector>
#include<string>
#include<list>
#include<exception>
#include<stdexcept>

using namespace std;

#define neuron_value(group, neuron, state) group->S[neuron+state*(group->num_neurons)]

#define BrianException std::runtime_error

class SpikeQueue
{
private:
public:
  SpikeQueue(int n0, int n1);
  ~SpikeQueue();
  
  long **X, *n;
  int n_delays, n_maxevents, currenttime, something;
  long *retarray;

  void expand();

  void next();

  void _peek(int nevents);

  // my two attempts at returning a numpy array
  void peek(long **ret, int *ret_n);
  void peek2(long *ret_out, int ret_n_out);

  void minimal();

  void insert(long delay[], long target[], int nevents);

  /* string __repr__(); */
  /* string __str__(); */
  /* //printout c++ */
  void print_summary();
};


#endif

