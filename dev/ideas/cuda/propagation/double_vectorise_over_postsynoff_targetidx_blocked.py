'''
Notes:

This algorithm is based on Issam's forDan/283_end.cu.

The idea (from Romain) is to do basically the same thing as in
vectorise_over_postsynaptic_offset but we do a double vectorisation. In the
first pass, the vectorisation is over the postsynaptic offset, but this time
we accumulate into a shared memory buffer without atomics (but this SHOULD
be done atomically right? just in shared memory atomics which is nicer). After,
we first sync the threads, then we propagate the shared memory buffer into
global memory.

Note that in Issam's version there are two potential bugs:
1. shared memory propagation is done without atomics, but I think conflicts are
   still possible.
2. it only works if the number of postsynaptic neurons per presynaptic neuron
   is always less than the shared memory size.

Performance:
- everything is coalesced except the shared memory atomics, but these are not
too costly. On the other hand, we potentially waste a lot of time
propagating unnecessary from shared to global memory. This might be fixed by
having warp-sized boolean flags to say whether or not a propagation is needed.
'''
