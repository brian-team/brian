from pylab import *
import glob, pickle, datetime

names = glob.glob('track_*.py')

def run_speedtracks(doplot=True):
    for name in names:
        mod = __import__(name.replace('.py',''))
        print 'Running tracking from', name
        descname, results = mod.run_barrage()
        if doplot:
            figure()
            mod.plot_results(descname, results)
        dateandtime = datetime.datetime.today()
        date = datetime.date.today()
        fname = name.replace('.py', '-'+str(date)+'.pkl')
        pickle.dump((descname, results, date), open(fname, 'wb'), protocol=2)
#        date = date - datetime.timedelta(1)
#        fname = name.replace('.py', '-'+str(date)+'.pkl')
#        pickle.dump((descname, results, date), open(fname, 'wb'), protocol=2)
    if doplot:
        show()

def plot_recent(name=None):
    if name is None:
        for name in names:
            plot_recent(name)
        return
    name = name.replace('.py','')
    mod = __import__(name)
    fnames = glob.glob(name+'-*.pkl')
    if len(fnames)==0:
        return
    bestdate = None
    bestdescname, bestresults = None, None
    for fname in fnames:
        descname, results, date = pickle.load(open(fname, 'rb'))
        if bestdate is None or date>bestdate:
            bestdescname, bestresults, bestdate = descname, results, date
    figure()
    mod.plot_results(bestdescname, bestresults)

def plot_all(name=None):
    if name is None:
        for name in names:
            plot_all(name)
        return
    name = name.replace('.py','')
    mod = __import__(name)
    fnames = glob.glob(name+'-*.pkl')
    for fname in fnames:
        descname, results, date = pickle.load(open(fname, 'rb'))
        figure()
        mod.plot_results(descname, results)

def meanresult(results):
    return sum(t for d, t, a in results)/len(results)

def track_mean(name=None):
    if name is None:
        for name in names:
            track_mean(name)
        legend()
        return
    name = name.replace('.py','')
    mod = __import__(name)
    fnames = glob.glob(name+'-*.pkl')
    bestdate = None
    for fname in fnames:
        descname, results, date = pickle.load(open(fname, 'rb'))
        if bestdate is None or date>bestdate:
            bestdate = date
    means = []
    for fname in fnames:
        descname, results, date = pickle.load(open(fname, 'rb'))
        means.append(((date-bestdate).days, meanresult(results)))
    means.sort()
    meanday, meanres = zip(*means)
    plot(meanday, meanres, label=descname)
        

if __name__=='__main__':
    run_speedtracks()
    #plot_recent()
    #track_mean()
    show()