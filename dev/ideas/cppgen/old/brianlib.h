#ifndef _BRIAN_LIB_H
#define _BRIAN_LIB_H

#include<vector>
#include<string>
using namespace std;

class LabelledArrays
{
	string a;
	vector<double> b;
public:
	LabelledArrays(string a, vector<double> b);
	string get_msg();
};

class LabelledArrays2
{
	string a;
	double *b;
	int n;
public:
	LabelledArrays2(string a, double *b, int n);
	string get_msg();
};

#endif 
