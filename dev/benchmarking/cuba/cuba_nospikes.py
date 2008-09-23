import brian_no_units
from brian import *
set_global_preferences(useweave=True)
import cuba_runopts
cuba_runopts.cuba_opts['connections'] = False
import cuba
cuba.do_runs('data/brian_cuba_nospiking_results_compile_nounits.pkl')
