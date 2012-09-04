#include "spikequeue.h"
#include<string.h>
#include<iostream>

using namespace std;

SpikeQueue::SpikeQueue(int n_delays, int n_maxevents)
{
  this->n_delays = n_delays;
  this->n_maxevents = n_maxevents;
  
  this->retarray = NULL;
  this->retarray = new long[n_maxevents];

  this->X = new long*[n_delays];
  for (int i = 0 ; i < n_delays ; i++)
    {
      (this->X)[i] = new long[n_maxevents];
    }

  this->n = new long[n_delays];
  this->currenttime = 0;
  
  // memory use control (copied from the ccircular.cpp)
  if(!this->X || !this->n){
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
    //    throw BrianException("Not enough memory in creating SpikeQueue.");
  }
}

// Spike Queue data structure
void SpikeQueue::expand()
{
   int orig_n_maxevents = this->n_maxevents;
   this->n_maxevents += orig_n_maxevents; // we multiply by 2.

   long *new_retarray = new long[this->n_maxevents];
   long **new_X = new long*[this->n_delays];
   for (int i = 0 ; i < n_delays ; i++)
     {
     new_X[i] = new long[this->n_maxevents];
     memcpy((void *)new_X[i], (void *)(this->X[i]), sizeof(long)*(this->n)[i]);
     }
   cout << "Done" << endl;
   if(!new_X){
     if(new_X) delete [] new_X;
     //     throw BrianException("Not enough memory in expanding SpikeQueue.");
   }
   delete [] this->X;
   this->X = new_X;
   delete [] this->retarray;
   this->retarray = new_retarray;
}

void SpikeQueue::next()
{
  this->n[this->currenttime] = 0; // erase
  this->currenttime = (this->currenttime + 1) % (this->n_delays);
}

void SpikeQueue::_peek(int nevents)
{
  //  
  //  cout << "nevents: " << nevents;
  for (int i = 0; i<nevents; i++)
    {
      this->retarray[i] = (this->X)[this->currenttime][i];
    }

}

// my two attempts at returning a numpy array
void SpikeQueue::peek(long **ret, int *ret_n)
{
  // This is where it fails for now.
  cout << "OK";
  int nevents = (this->n)[this->currenttime];
  cout << "OK";
  this->_peek(nevents);
  *ret = (this->retarray);
  *ret_n = nevents;
}
void SpikeQueue::peek2(long *ret_out, int ret_n_out)
{
  // This is where it fails for now.
  int nevents = (this->n)[this->currenttime];
  this->_peek(nevents);
  ret_out = (this->retarray);
  ret_n_out = nevents;
}


void SpikeQueue::insert(long delay[], long target[], int nevents)
{
  //  int nevents = sizeof(delay)/sizeof(long);
  //  cout << "nevents: " << nevents;
  for (int k = 0; k < nevents; k ++)
    {
      const int d = (this->currenttime + delay[k]) % (this->n_delays);
      (this->X)[d][(this->n)[d]] = target[k];

      (this->n)[d]++;
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

///////////////////// MAIN ///////////////////////
/*
Here I do some testing.
 */
int main(void){
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
  x.insert(delay, target, 4);

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

  return 1;
}
