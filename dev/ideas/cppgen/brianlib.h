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

#endif 
