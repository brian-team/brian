from numpy import *

__all__ = ['apply_batched']

def apply_batched_orig(op, inds, debug=False):
    '''
    Applies op(tgt, src) in batches, where tgt will have each of the values in
    inds precisely once, and src will cover each integer from 0 to len(inds)-1
    precisely once. For each call op(tgt, src) tgt and src are arrays of
    indices where tgt=inds[src], and it is guaranteed that there are
    no repeated values in tgt. 
    '''
    u, i = unique(inds, return_index=True)
    if debug:
        print inds, u, i
    op(u, i)
    if len(u)<len(inds):
        inds[i] = -1
        while len(u)<len(inds) and (inds>-1).any():
            u, i = unique(inds, return_index=True)
            if debug:
                print inds, u, i
            op(u[1:], i[1:])
            inds[i] = -1
            

def apply_batched_orig_improved(op, inds, debug=False):
    '''
    Applies op(tgt, src) in batches, where tgt will have each of the values in
    inds precisely once, and src will cover each integer from 0 to len(inds)-1
    precisely once. For each call op(tgt, src) tgt and src are arrays of
    indices where tgt=inds[src], and it is guaranteed that there are
    no repeated values in tgt. 
    '''
    u, i = unique(inds, return_index=True)
    if debug:
        print inds, u, i
    op(u, i)
    if len(u)<len(inds):
        I = ones(len(inds), dtype=bool)
        I[i] = False
        remaining, = I.nonzero()
        while 1:
            indsrem = inds[remaining]
            u, i = unique(indsrem, return_index=True)
            if debug:
                print inds, u, i
            op(u, remaining[i])
            if len(u)==len(indsrem):
                break
            I = ones(len(remaining), dtype=bool)
            I[i] = False
            remaining = remaining[I]


def apply_batched(op, inds, debug=False):
    '''
    Applies op(tgt, src) in batches, where tgt will have each of the values in
    inds precisely once, and src will cover each integer from 0 to len(inds)-1
    precisely once. For each call op(tgt, src) tgt and src are arrays of
    indices where tgt=inds[src], and it is guaranteed that there are
    no repeated values in tgt.
    '''
    perm = inds.argsort()
    aux = inds[perm]
    flag = empty(len(aux)+1, dtype=bool)
    flag[0] = flag[-1] = True
    not_equal(aux[1:], aux[:-1], flag[1:-1])
    F = flag.nonzero()[0][:-1]
    logical_not(flag, flag)
    if debug:
        print '\n'.join(map(str, [inds, perm, aux, array(flag, dtype=int)]))+'\n'
    while len(F):
        u = aux[F]
        i = perm[F]
        if debug:
            print u, i, F
        op(u, i)
        F += 1
        F = F[flag[F]]

                       
if __name__=='__main__':
    if 1:
        x = ones(10)
        inds = array([7, 8, 9, 9, 8, 8])
        def op(u, i):
            x[u] = x[u]**2+1
            return x
        
        apply_batched(op, inds, debug=True)
        
        print x
        print

#    exit(0)
        
    if 1:
        ids = array([0, 1, 1, 2])
        n = array([ [0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1], [0.1, 0.1, 0.1] ])
        nv = zeros((3, 3))
        def op(tgt, src):
#            print tgt, src, ids[src]
#            print tgt==ids[src]
            nv[tgt] += n[src]
        apply_batched(op, ids.copy())
        print nv
        print
        
    if 1:
        import time
        from numpy.random import *
        from scipy import weave
        ids = randint(1000, size=10000)
        repeats = 100
        
        y = randn(len(ids))
        
        x_loop = zeros(amax(ids)+1)
        start = time.time()
        for _ in xrange(repeats):
            ids.copy()
            for i in xrange(len(ids)):
                x_loop[ids[i]] += y[i]
        print 'Loop:', time.time()-start

        x_weave = zeros(amax(ids)+1)
        numinds = len(ids)
        start = time.time()
        for _ in xrange(repeats):
            ids.copy()
            weave.inline('''
                for(int i=0; i<numinds; i++)
                    x_weave[ids[i]] += y[i];
                ''',
                ['x_weave', 'y', 'ids', 'numinds'],
                compiler='gcc',
                extra_compile_args=['-O3', '-march=native'],
                )
        print 'Weave:', time.time()-start,
        print 'diff', amax(abs(x_loop-x_weave))#<1e-10

        def dobatchtiming(batchfunc):
            global numops
            numops = 0
            x_batched = zeros(amax(ids)+1)
            def op(tgt, src):
                global numops
                numops += 1
                x_batched[tgt] += y[src]
            start = time.time()
            for _ in xrange(repeats):
                batchfunc(op, ids.copy())
            print batchfunc.__name__, ':', time.time()-start, '(%d ops)'%(numops/repeats),
            print 'diff', amax(abs(x_loop-x_batched))
            
        dobatchtiming(apply_batched_orig)
        dobatchtiming(apply_batched_orig_improved)
        dobatchtiming(apply_batched)
