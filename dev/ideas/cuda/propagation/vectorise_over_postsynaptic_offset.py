'''
Notes:

Based on Issam's forDan/178_end.cu.

The idea is that for each spiking neuron i, it is connected to tgt[off[i]+k] for
0<=k<numsynapses[i], and we vectorise over k. This looks something like this if
neurons 0, 3 and 4 spiked:

i    tgt[off[i]] ...        numsynapses[i]
0    2  4  5  8  15         5
3    3  9                   2
4    0  1  2                3

We launch max(numsynapses) threads, so we waste time if the structure is very
ragged, but is relatively efficient if it is uniform (i.e. numsynapses[i]
const).

The kernel looks like this:
    __global__ ...
        int thread = ...
        targets = tgt_arr+off[i];
        weights = w_arr+off[i];
        for(i=0;i<numspikes;i++)
            if(i<numsynapses[i])
                j = targets[thread]; // coalesced read
                w = weights[thread]; // coalesced read
                atomic V[j]+=w;      // uncoalesced read/write

Theoretical analysis of performance:
- read of targets and weights is nicely coalesced
- if numsynapses~10000 then we are using plenty of threads, efficient use
- writing to target state variable is uncoalesced, inefficient
    + means that we typically have long waits for each synapse, must be
      inefficient?
- writing to target state variable is atomic, inefficient
    + this cost depends on the number of conflicts, but in many cases may not
      be too bad
'''