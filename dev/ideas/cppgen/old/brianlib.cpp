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

LabelledArrays2::LabelledArrays2(string a, double *b, int n)
{
	this->a = a;
	this->b = b;
	this->n = n;
}

string LabelledArrays2::get_msg()
{
	double x = 0.0;
	for(int i=0;i<this->n;i++)
		x += this->b[i];
	string s;
	stringstream out;
	out << "array = ";
	for(int i=0;i<this->n;i++)
		out << this->b[i] << " ";
	out << endl;
	out << "sum of array labelled " << this->a << " of length " << this->n << " is " << x;
	s = out.str();
	return s;
}
