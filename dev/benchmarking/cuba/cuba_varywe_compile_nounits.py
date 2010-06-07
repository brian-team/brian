import brian_no_units
from brian import *
set_global_preferences(useweave=True)
import cuba
from cuba_runopts import N_varywe
cuba.do_runs_varywe('data/brian_cuba_varywe_results_compile_nounits_'+str(N_varywe)+'.pkl')
