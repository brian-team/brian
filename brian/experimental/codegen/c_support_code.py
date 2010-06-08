c_support_code = '''
double clip(double x, double low, double high)
{
    if(x<low) return low;
    if(x>high) return high;
    return x;
}
#define inf INFINITY
'''
