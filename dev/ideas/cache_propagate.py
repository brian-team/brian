'''
Seems to make it go slower rather than faster...
'''

from brian import *
import numpy
from scipy import weave

neuron_column_width = 512 # various values, none of them make it faster

class CacheConnection(Connection):
    def propagate(self, spikes):
        if not self.iscompressed:
            self.compress()
        if len(spikes):
            if not isinstance(spikes, numpy.ndarray):
                spikes = array(spikes, dtype=int)
            sv = self.target._S[self.nstate]
            rows = self.W.get_rows(spikes)
            nspikes = len(spikes)
            N = len(sv)
            code = """
                    using namespace std;
                    //vector< blitz::Array<double,1> > rowset;
                    vector< double* > rowset;
                    rowset.reserve(nspikes);
                    for(int j=0;j<nspikes;j++)
                    {
                        PyObject* _rowsj = rows[j];
                        PyArrayObject* _row = convert_to_numpy(_rowsj, "row");
                        conversion_numpy_check_type(_row, PyArray_DOUBLE, "row");
                        conversion_numpy_check_size(_row, 1, "row");
                        //blitz::Array<double,1> row = convert_to_blitz<double,1>(_row,"row");
                        //rowset[j] = row;
                        rowset[j] = (double*)_row->data;
                        Py_DECREF(_rowsj);                        
                    }
                    for(int start=0; start<N; start+=neuron_column_width)
                    {
                        int end = start+neuron_column_width;
                        if(end>N) end=N;
                        for(int j=0;j<nspikes;j++)
                        {
                            /*PyObject* _rowsj = rows[j];
                            PyArrayObject* _row = convert_to_numpy(_rowsj, "row");
                            conversion_numpy_check_type(_row, PyArray_DOUBLE, "row");
                            conversion_numpy_check_size(_row, 1, "row");
                            blitz::Array<double,1> row = convert_to_blitz<double,1>(_row,"row");*/
                            //blitz::Array<double,1> row = rowset[j];
                            double *row = rowset[j];
                            for(int k=start;k<end;k++)
                                //sv(k) += row(k);
                                sv(k) += row[k];
                            //Py_DECREF(_rowsj);
                        }
                    }
                    """
            weave.inline(code, ['sv', 'spikes', 'nspikes', 'N', 'rows', 'neuron_column_width'],
                         compiler=self._cpp_compiler,
                         type_converters=weave.converters.blitz,
                         extra_compile_args=self._extra_compile_args,
                         headers=['<vector>'])

if __name__=='__main__':

    log_level_debug()

    import time
    
    #set_global_preferences(usenewpropagate=False)
    
    N = 10000
    tau = 10*ms
    duration = 100*ms
    p = 0.1
    
    eqs = '''
    dv/dt = -v/(10*ms) : 1
    '''
    
    G = NeuronGroup(N, eqs, reset=-2, threshold=-1)
    G.v = rand(N)-2
    
    C = Connection(G, G, 'v', weight=1e-10, sparseness=p, structure='dense')
    #C = CacheConnection(G, G, 'v', weight=1e-10, sparseness=p, structure='dense')
    
    M = SpikeMonitor(G, record=False)
    
    run(1*ms)

    forget(C)
    start = time.time()
    run(duration)
    end = time.time()
    t_noconnection = end-start
    recall(C)    
    
    start = time.time()
    run(duration)
    end = time.time()
    t_both = end-start
    t_connection = t_both - t_noconnection
    
    print 'Propagation time:', t_connection
    print 'Average neuron firing rate:', M.nspikes/duration/N
    print 'Average spikes per time step:', M.nspikes/(duration/defaultclock.dt)
    print 'Average synapses per time step:', M.nspikes*(N*p)/(duration/defaultclock.dt)
    
#    raster_plot(M)
#    show()
