import brian_no_units
from brian import *
set_global_preferences(useweave=True)
import cuba
cuba.do_runs('data/brian_cuba_results_compile_nounits.pkl')
#print cuba.cuba_average(32000,1,1)
