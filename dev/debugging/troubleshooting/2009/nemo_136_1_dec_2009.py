from brian import *

N=200

defaultclock.dt = 0.5*ms

izh_eqs='''
dv/dt=0.04/ms/mV*v**2+5/ms*v+140*mV/ms-u+ge-gi+i:volt
du/dt=a*(b*v-u):volt/second
a:1/second
b:1/second
dge/dt=-ge/ms:volt/second
dgi/dt=-gi/ms:volt/second
i:volt/second
c:volt
d:volt
'''


reset_eqs='''
v=c;u=u+d
'''
group = NeuronGroup(200, model=izh_eqs, threshold=30*volt, reset=reset_eqs)

sub1=group[:100]
sub2=group[100:]

for i in range(len(sub1)):
  sub1[i].a=0.02/ms
  sub1[i].b=0.2/ms
  sub1[i].c=-65.0*mV
  sub1[i].d=8.0*mV

for i in range(len(sub2)):
  sub2[i].a=0.02/ms
  sub2[i].b=0.2/ms
  sub2[i].c=-65.0*mV
  sub2[i].d=8.0*mV


#direct links
linksexcit=Connection(sub1,sub2,'ge',structure='dense')

#CASE 1
linksexcit.connect_full(sub1,sub2,weight=lambda i,j:rand()*10.0*mV/ms)
#linksexcit.connect_full(sub1,sub2,weight=lambda i,j:rand(array(j).size)*10.0*mV/ms)

#CASE 2
#create custom links
#for i in range(len(sub1)):
#  for j in range(len(sub2)):
#    linksexcit[i,j]=rand()*10*mV/ms

clock_i=Clock(dt=1*ms)
@network_operation
def myoperation(clock_i):
    sub1.i=[10*randn()*mV/ms for i in range(len(sub1))]

group.v=-65*mV
group.u=group.v*group.b


spkm=SpikeMonitor(group)
spkc1=SpikeCounter(sub1)
spkc2=SpikeCounter(sub1)

run(200*ms)
print spkc1.nspikes
print spkc2.nspikes
raster_plot()
figure()
imshow(linksexcit.W.todense())
colorbar()
show() 
