/* #include<vector> */
/* #include<string> */
/* #include<list> */
/* #include<iostream> */
//using namespace std;

class SpikeQueue
{
private:
public:
  SpikeQueue(int n0, int n1);

  long **X, *n;
  int n_delays, n_maxevents, currenttime;
  long *retarray;

  void expand();

  void next();

  void _peek(int nevents);
  void peek(long **ret, int *ret_n);
  void insert(long delay[], long target[], int nevents);

  //printout c++
  void print_summary();
};

