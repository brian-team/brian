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
  
  long **X;
  int *n;
  int n_delays, n_maxevents, currenttime, something;
  long *retarray;

  void expand(int maxevents);

  void next();

  void _peek(int nevents);

  void peek(long **ret, int *ret_n);

  void insert(int len1, long *vec1, int len2, long *vec2);
  
  string __repr__(); /* doesn't work, but doesn't work in ccircular either. */
  string __str__();
  /* //printout c++ */

  void print_summary();
};


#endif

