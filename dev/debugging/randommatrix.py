from brian import *

P=NeuronGroup(100, model='dv/dt=0*Hz:1')
Q=NeuronGroup(200, model='dv/dt=0*Hz:1')
C=Connection(P, Q, 'v')
C.connect_random(P, Q, sparseness=lambda i, j:exp(-.05*abs(i-j)), weight=lambda i, j:abs(cos(.1*(i-j))))
C2=Connection(P, Q, 'v')
C2.connect_random(P, Q, sparseness=lambda i, j:exp(-.05*abs(i-j)), weight=lambda: rand())

print C.W.nnz, C2.W.nnz

figure()
subplot(211)
imshow(C.W.todense(), interpolation='nearest', origin='lower')
subplot(212)
imshow(C2.W.todense(), interpolation='nearest', origin='lower')
show()
