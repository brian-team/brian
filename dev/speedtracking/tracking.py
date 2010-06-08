import time

def track(t):
    # could do something more complicated here to average over several runs
    start = time.time()
    t.run()
    return time.time() - start
