#include "spikequeue.h"
#include<string.h>
#include<iostream>
#include<sstream>

using namespace std;

SpikeQueue::SpikeQueue(int n_delays, int n_maxevents)
{
  this->n_delays = n_delays; // first dimension of X (nsteps)
  this->n_maxevents = n_maxevents; // second dimension of X 

  this->currenttime = 0;
  
  this->retarray = NULL;

  this->something = -1;

  this->retarray = new long[n_maxevents];
  this->n = new int[n_delays];
  this->X = new long*[n_delays];

  if(!this->X || !this->n || !this->retarray){
    if(this->X) {
      delete [] this->X;
      this->X = 0;
    }
    if(this->n) {
      delete [] this->n;
      this->n = 0;
    }
    if(this->retarray) {
      delete [] this->retarray;
      this->retarray = 0;
    }
    throw BrianException("Not enough memory in creating SpikeQueue.");
  }

  for (int i = 0 ; i < n_delays ; i++)
    {
      (this->X)[i] = new long[n_maxevents];
      (this->n)[i] = 0;
      for (int j = 0 ; j < n_maxevents; j++)
	{
	  (this->X)[i][j] = 0;
	}
      if (!((this->X)[i]))
	{
	  throw BrianException("Not enough memory in creating SpikeQueue (X).");
	}
    }
}
SpikeQueue::~SpikeQueue()
{
  for (int i = 0 ; i < this->n_delays ; i++)
    {
      delete [] (this->X)[i];
    }
  if(this->X) delete [] this->X;
  if(this->retarray) delete [] this->retarray;
  if(this->n) delete [] this->n;
  
  this->X = NULL;
  this->retarray = NULL;
  this->n = NULL;
  
}
// Spike Queue data structure
void SpikeQueue::expand(int maxevents = -1)
{
  if (maxevents != -1){
    this->n_maxevents += maxevents; // we add maxevents
  }
  else
    {
      int orig_n_maxevents = this->n_maxevents;
      this->n_maxevents += orig_n_maxevents; // we multiply by 2.
    }
  
  // declare and allocate
  long *new_retarray = new long[this->n_maxevents];
  long **new_X = new long*[this->n_delays];
  
  // expand the 2D array structure
  for (int i = 0 ; i < n_delays ; i++)
    {
      // allocate new memory
      new_X[i] = new long[this->n_maxevents];
       // check allocation success
      if (!(new_X[i]))
	{
	  stringstream err;
	  err << "Not enough memory in expanding SpikeQueue to size (";
	  err << this->n_delays << ",";
	  err << this->n_maxevents << ")";
	  throw BrianException(err.str());
	}
       // copy old contents of X
      memcpy((void *)new_X[i], (void *)(this->X[i]), sizeof(long)*(this->n)[i]);
      // delete old contents
      delete [] (this->X)[i];
      // zero out newly allocated data
      for (int j = (this->n)[i] ; j < this->n_maxevents; j++)
	{
	  new_X[i][j] = 0;
	}
    }
  
  if(!new_X || !retarray){
    if(new_X) delete [] new_X;
    throw BrianException("Not enough memory in expanding SpikeQueue.");
  }
  // delete 
  delete [] this->X;
  // assign
  this->X = new_X;
  // re-delete
  delete [] this->retarray;
  // re-assign
  this->retarray = new_retarray;
}

void SpikeQueue::next()
{
  this->n[this->currenttime] = 0; // erase
  this->currenttime = ((this->currenttime + 1) % (this->n_delays));
}

void SpikeQueue::_peek(int nevents)
{
  for (int i = 0; i < nevents; i++)
    {
      this->retarray[i] = (this->X)[this->currenttime][i];
    }
}
// my two attempts at returning a numpy array
void SpikeQueue::peek(long **ret, int *ret_n)
{
  int nevents = (this->n)[this->currenttime];
  this->_peek(nevents);
  *ret = this->retarray;
  *ret_n = nevents;
}

void SpikeQueue::insert(int len1, long *vec1, int len2, long *vec2)
{
  // vec1 = targets
  // vec2 = delays
  if (len1!=len2) 
    {
      throw BrianException("Inputs to insert have non matching lengths");
    }
  for (int k = 0; k < len1; k ++)
    {
      // get the timebin of this spike
      int d = (this->currenttime + vec2[k]) % (this->n_delays);
      if ((this->n)[d] == this->n_maxevents)
	{
	  this->expand();
	}
      // place it in the 2D array
      (this->X)[d][(this->n)[d]] = vec1[k];
      (this->n)[d]++;
      // check that we can fit the events

    }
}

// DEBUG/PRINTING
void SpikeQueue::print_summary()
{
  cout << "SpikeQueue" << endl;
  cout << "n_maxevents: " << this->n_maxevents << endl;
  cout << "n_delays: " << this->n_delays << endl;
  cout << "currenttime: " << this->currenttime << endl;
  cout << "Contents" << endl;
  
  for (int i = 0; i < (this->n_delays); i ++){
    cout << '(' << (this->n)[i] << ')';
    for (int j = 0; j < this->n[i]; j ++){
      cout << (this->X)[i][j] << ',';
    }
    cout << endl;
  }
}
// Even this simple thing doesnt work.
string SpikeQueue::__repr__()
{
  stringstream out;
  out << "SpikeQueue" << endl;
  out << "n_delays = " << this->n_delays << endl;
  out << "n_maxevents = " << this->n_maxevents << endl;
  out << "currenttime = " << this->currenttime << endl;
  int n = 0;
  for (int k=0; k<this->n_delays; k++){
    n += (this->n)[k];
    out << "(" << k << ")" << (this->n)[k] << endl;
  }
  out << "Contains " << n << " spikes" << endl;
  return out.str();
}
string SpikeQueue::__str__()
{
  return this->__repr__();
}


///////////////////// MAIN ///////////////////////

int main(void){
  /*
  int N = 5;
  SpikeQueue x (10, N);

  cout << "////////////////" << endl;
  cout << "///// INIT /////" << endl;
  cout << "////////////////" << endl;

  x.print_summary();

  cout << "////////////////" << endl;
  cout << "Inserting spikes" << endl;
  cout << "////////////////" << endl;
  
  long delay[4] = {1, 2, 2, 3};
  long target[4] = {45, 45, 46, 46};
  //  x.insert(delay, target, 4);

  cout << "////////////////" << endl;
  cout << "///// CHECK /////" << endl;
  cout << "////////////////" << endl;
  // We should see some spikes
  x.print_summary();

  cout << "////////////////" << endl;
  cout << "///// NEXT /////" << endl;
  cout << "////////////////" << endl;
  // We should see currenttime = 1
  x.next();
  x.print_summary();

  cout << "////////////////" << endl;
  cout << "///// EXPND ////" << endl;
  cout << "////////////////" << endl;
  x.expand();
  // We should not see any difference
  x.print_summary();
  
  cout << "////////////////" << endl;
  cout << "///// PEEK /////" << endl;
  cout << "////////////////" << endl;
  x._peek(N);
  // I have to print manually, we should see the numbers corresponding to the right line in the data.
  for (int i = 0; i < N; i++)
    {
      cout << x.retarray[i] << ',';
    }
  cout << endl;

  cout << "////////////////" << endl;
  cout << "/ NEXT + PEEK //" << endl;
  cout << "////////////////" << endl;
  x.next();
  x._peek(N);
  for (int i = 0; i < N; i++)
    {
      cout << x.retarray[i] << ',';
    }

  cout << endl;
  cout << "////////////////" << endl;
  cout << "/ NEXT + PEEK 2/" << endl;
  cout << "////////////////" << endl;
  x.next();
  x._peek(N);
  for (int i = 0; i < N; i++)
    {
      cout << x.retarray[i] << ',';
    }
  cout << endl;

  int n;
  long *ret;
  x.peek(&ret, &n);
  
  cout << "////////////////" << endl;
  cout << "/ LAST /" << endl;
  cout << "////////////////" << endl;

  for (int i = 0; i < n; i++)
    {
      cout << ret[i] << ',';
    }
  cout << endl;

  return 1;
*/
}

