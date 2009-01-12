from pylab import *
import glob, pickle

names = glob.glob('track_*.py')

for name in names:
    mod = __import__(name.replace('.py',''))
    print 'Running tracking from', name
    name, results = mod.run_barrage()
    mod.plot_results(name, results)

show()
