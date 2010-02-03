extern "C" {

__declspec(dllexport) double dosum(double *x, int n)
{
	double ret=0.0;
	for(int i=0;i<n;i++)
		ret += x[i];
	return ret;
}

}
