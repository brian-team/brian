import brian_no_units
from brian import *
from filter import *
from scipy.optimize import *

class ElectrodeCompensation (object):
    eqs = """
            dV/dt=Re*(-Iinj)/taue : volt
            dV0/dt=(R*Iinj-V0+Vr)/tau : volt
            Iinj=(V-V0)/Re : amp
            """

    def __init__(self, I, Vraw,
                 dt=defaultclock.dt, durslice=1*second,
                  p=1.0, 
                 *params):
        self.I = I
        self.Vraw = Vraw
        self.p = p
        self.dt = dt
        self.x0 = self.params_to_vector(*params)
        self.duration = len(I) * dt
        self.durslice = min(durslice, self.duration)
        self.slicesteps = int(durslice/dt)
        self.nslices = int(ceil(len(I)*dt/durslice))
        
        self.islice = 0
        self.I_list = [I[self.slicesteps*i:self.slicesteps*(i+1)] for i in range(self.nslices)]
        self.Vraw_list = [Vraw[self.slicesteps*i:self.slicesteps*(i+1)] for i in range(self.nslices)]

    def vector_to_params(self, *x):
        R,tau,Vr,Re,taue = x

        R = R*R
        tau = tau*tau
        Re = Re*Re
        taue = taue*taue

        return R,tau,Vr,Re,taue

    def params_to_vector(self, *params):
        x = params

        x = [sqrt(params[0]),
             sqrt(params[1]),
             params[2],
             sqrt(params[3]),
             sqrt(params[4])]

        return list(x)

    def get_model_trace(self, row, *x):
        R, tau, Vr, Re, taue = self.vector_to_params(*x)
        eqs = Equations(self.eqs)
        eqs.prepare()
        self._eqs = eqs
        y = simulate(eqs, self.I_list[self.islice] * Re/taue, self.dt, row=row)
        return y

    def fitness(self, x):
        R, tau, Vr,  Re, taue = self.vector_to_params(*x)
        y = self.get_model_trace(0, *x)
        e = self.dt*sum(abs(self.Vraw_list[self.islice]-y)**self.p)
        return e

    def compensate_slice(self, x0):
        fun = lambda x: self.fitness(x)
        x = fmin(fun, x0, maxiter=10000, maxfun=10000)
        return x

    def compensate(self):
        self.params_list = []
        self.xlist = [self.x0]
        t0 = time.clock()
        for self.islice in range(self.nslices):
            newx = self.compensate_slice(self.xlist[self.islice])
            self.xlist.append(newx)
            self.params_list.append(self.vector_to_params(*newx))
            print "Slice %d/%d compensated in %.2f seconds" %  \
                (self.islice+1, self.nslices, time.clock()-t0)
            t0 = time.clock()
        self.xlist = self.xlist[1:]
        return self.xlist

    def get_compensated_trace(self):
        Vcomp_list = []
        Vmodel_list = []
        for self.islice in range(self.nslices):
            x = self.xlist[self.islice]
            V = self.get_model_trace(0, *x)
            V0 = self.get_model_trace(1, *x)
            Velec = V-V0
            Vcomp_list.append(self.Vraw_list[self.islice] - Velec)
            Vmodel_list.append(V)
        self.Vcomp = hstack(Vcomp_list)
        self.Vmodel = hstack(Vmodel_list)
        return self.Vcomp, self.Vmodel

def compensate(I, Vraw, 
               dt=defaultclock.dt, 
               durslice=1*second,
               p=1.0,
               **initial_params):

    R = initial_params.get("R", 100*Mohm)
    tau = initial_params.get("tau", 20*ms)
    Vr = initial_params.get("Vr", -70*mV)
    Re = initial_params.get("Re", 50*Mohm)
    taue = initial_params.get("taue", .5*ms)

    comp = ElectrodeCompensation(I, Vraw,
                                 dt,
                                 durslice,
                                 p,
                                 R, tau, Vr, Re, taue)
    comp.compensate()
    Vcomp, Vmodel = comp.get_compensated_trace()
    params = comp.params_list
    return Vcomp, Vmodel, params


if __name__ == '__main__':

    I = numpy.load("current1.npy")[:100000] * 80
    Vraw = numpy.load("trace1.npy")[:100000]

    R = 200*Mohm
    tau = 20*ms
    Vr = -70*mV
    Re = 50*Mohm
    taue = .5*ms

    Vcomp, Vmodel, params = compensate(I, Vraw, dt=.1*ms,
                                       p = 1.0,
                                        durslice=10*second,
                                        R=R, tau=tau, Vr=Vr, Re=Re, taue=taue)

    #subplot(211)
    #plot(I)

    #subplot(212)
    plot(Vraw, 'k')
    plot(Vmodel, 'g')
    plot(Vcomp, 'r')

    show()
