#include "brianlib.h" 
#include<sstream>
using namespace std;

LabelledArrays::LabelledArrays(string a, vector<double> b)
{
	this->a = a;
	this->b = b;
}

string LabelledArrays::get_msg()
{
	double x = 0.0;
	for(vector<double>::iterator i=this->b.begin();i!=this->b.end();i++)
		x += *i;
	string s;
	stringstream out;
	out << "sum of array labelled " << this->a << " is " << x;
	s = out.str();
	return s;
}
