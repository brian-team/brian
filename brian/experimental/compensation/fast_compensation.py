import brian_no_units
from brian import *
from filter import *
from scipy.optimize import *




class ElectrodeCompensation (object):


    eqs = """
            dV/dt=Re*(-Iinj)/taue : volt
            dV0/dt=(R*Iinj-V0+Vr)/tau : volt
            Iinj=(V-V0)/R : amp
            """

    # coefficients before I for each variable
    U = lambda self, Re, taue: \
            array([ Re/taue, 0., 0.])

    def __init__(self, I, Vraw, p=2.0, 
                 dt=defaultclock.dt, durslice=1*second,
                 *params):
        self.I = I
        self.Vraw = Vraw
        self.p = p
        self.dt = dt
        self.x0 = self.params_to_vector(*params)
        self.durslice = durslice
        self.slicesteps = int(durslice/dt)
        self.nslices = int(ceil(len(I)*dt/durslice))
        
        self.islice = 0
        self.I_list = [I[self.slicesteps*i:self.slicesteps*(i+1)] for i in range(self.nslices)]
        self.Vraw_list = [Vraw[self.slicesteps*i:self.slicesteps*(i+1)] for i in range(self.nslices)]



    def vector_to_params(self, *x):
        R,tau,Vr,Re,taue = x

        #R = R*R
        #tau = tau*tau
        #Re = Re*Re
        #taue = taue*taue

        return R,tau,Vr,Re,taue

    def params_to_vector(self, *params):
        x = params
        #x = [sqrt(params[0]),
        #     sqrt(params[1]),
        #     sqrt(params[2]),
        #     sqrt(params[3]),
        #     params[4]]
        return list(x)

    def get_model_trace(self, rows, U, *x):
        R, tau, Vr, Re, taue = self.vector_to_params(*x)
        eqs = Equations(self.eqs)
        eqs.prepare()

        self._eqs = eqs
        
        y = simulate(eqs, self.I, self.dt, rows=rows, U=U)
        return y

    def fitness(self, x):
        R, tau, Vr,  Re, taue = self.vector_to_params(*x)
        y = self.get_model_trace('all', self.U(Re, taue), *x)
        self.V = y[0,:]
        self.V0 = y[1,:]
        e = self.dt*sum(abs(self.Vraw_list[self.islice]-self.V)**self.p)
        return e

    def compensate_slice(self, x0):
        fun = lambda x: self.fitness(self.I_list[self.islice], x)
        x = fmin(fun, x0, maxiter=1000)
        return x

    def compensate(self):
        self.params_list = []
        xlist = [self.x0]
        for self.islice in range(self.nslices):
            print self.islice
            newx = self.compensate_slice(xlist[self.islice])
            xlist.append(newx)
            self.params_list.append(self.vector_to_params(*newx))
        return xlist[1:]

    def get_compensated_trace(self, xlist):
        Vmodel_list = []
        for i in range(self.nslices):
            x = xlist[i]
            Vmodel_list.append(self.get_model_trace(self.eqs_V, self.I_list[i], *x))
        return hstack(Vmodel_list)




def test_compensation():

    I = numpy.load("current1.npy")[:10000]
    Vraw = numpy.load("trace1.npy")[:10000]

    R = 1000*Mohm
    tau = 10*ms
    Vr = -70*mV
    Re = 60*Mohm
    taue = 1*ms


    t0 = time.clock()
    comp = ElectrodeCompensation(I, Vraw, 1, defaultclock.dt, 
                                 1*second,
                                 R, tau, Vr, Re, taue)
    xlist = comp.compensate()
    t1 = time.clock()-t0

    print t1
    Vmodel = comp.get_compensated_trace(xlist)


    subplot(211)
    plot(I)

    subplot(212)
    plot(Vraw)
    plot(Vmodel)

    show()

    


    
if __name__ == '__main__':
    test_compensation()
