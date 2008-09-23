import brian_no_units
from brian import *
set_global_preferences(useweave=True)
import cuba
cuba.do_runs_varyconnectivity('data/brian_cuba_varycon_results_compile_nounits.pkl')