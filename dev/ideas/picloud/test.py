from multimap import *
import time

def fun(x):
    return x**2

# max_parallel is defined in multimap.py and contains the maximum number of jobs that can
# run simultaneously.
args=range(max_parallel)

# Launch the jobs on the cloud and returns the job ids used to retrieve the results later.
jids=multimap(fun, args)

# Prints the status of the jobs after 10 seconds
time.sleep(10)
print status(jids)

# Retrieves the results (BLOCKING call)
results=retrieve(jids)
print results

