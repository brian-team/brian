from brian import *
import struct

x = array(rand(10)*1000, dtype = int16)
print 'x',x

x_n = x.newbyteorder('>')
print 'x_n',x_n

s_struct = ''
for v in x_n:
    s_struct += struct.pack('>i', v)   

s_np =x_n.tostring() 

if s_np == s_struct:
    print 'yay'
else:
    print s_np
    print s_struct



xre=fromstring(s_struct, dtype=int16) # or uint16?
xre = xre[:,0].newbyteorder('>')
print 'xre', xre

if (x == xre).all():
    print 'YAY'









