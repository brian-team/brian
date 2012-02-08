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

def apply_batched_take(op, inds, debug=False):
    '''
    Applies op(tgt, src) in batches, where tgt will have each of the values in
    inds precisely once, and src will cover each integer from 0 to len(inds)-1
    precisely once. For each call op(tgt, src) tgt and src are arrays of
    indices where tgt=inds[src], and it is guaranteed that there are
    no repeated values in tgt.
    '''
    perm = inds.argsort()
    aux = take(inds, perm)#inds[perm]
    flag = empty(len(aux)+1, dtype=bool)
    flag[0] = flag[-1] = True
    not_equal(aux[1:], aux[:-1], flag[1:-1])
    F = flag.nonzero()[0][:-1]
    logical_not(flag, flag)
    if debug:
        print '\n'.join(map(str, [inds, perm, aux, array(flag, dtype=int)]))+'\n'
    while len(F):
        u = take(aux, F)
        i = take(perm, F)
        if debug:
            print u, i, F
        op(u, i)
        F += 1
        #F = F[take(flag, F)]
        F = extract(take(flag, F), F)

def apply_batched_take2(op, inds, debug=False):
    '''
    Applies op(tgt, src) in batches, where tgt will have each of the values in
    inds precisely once, and src will cover each integer from 0 to len(inds)-1
    precisely once. For each call op(tgt, src) tgt and src are arrays of
    indices where tgt=inds[src], and it is guaranteed that there are
    no repeated values in tgt.
    '''
    perm = inds.argsort()
    aux = take(inds, perm)
    flag = empty(len(aux)+1, dtype=bool)
    flag[0] = flag[-1] = True
    not_equal(aux[1:], aux[:-1], flag[1:-1])
    if sum(flag)==len(aux)+1:
        op(inds, slice(None))
        return
    F = flag.nonzero()[0][:-1]
    logical_not(flag, flag)
    while len(F):
        u = take(aux, F)
        i = take(perm, F)
        op(u, i)
        F += 1
        F = extract(take(flag, F), F)
                       
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
        class AllIDs(object):
            def __init__(self, indmax, sz, repeats, frac_unique=1.0):
                self.indmax = indmax
                self.sz = sz
                self.repeats_notunique = int((1.0-frac_unique)*repeats)
                self.repeats_unique = repeats-self.repeats_notunique
                self.repeats = repeats
            def __iter__(self):
                indmax, sz = self.indmax, self.sz
                def f():
                    seed(304892)
                    for _ in xrange(self.repeats_unique):
                        yield randint(indmax, size=sz)
                    for _ in xrange(self.repeats_notunique):
                        yield arange(sz)
                return f()
        all_ids = AllIDs(indmax=100000,
                         sz=10000,
                         repeats=1000,
                         frac_unique=0.5
                         )
        do_loop = False
        
        seed(3402324)
        y = randn(all_ids.sz)
        
        if do_loop:
            x_loop = zeros(all_ids.indmax)
            start = time.time()
            for ids in all_ids:
                ids.copy()
                for i in xrange(len(ids)):
                    x_loop[ids[i]] += y[i]
            print 'Loop:', time.time()-start

        x_weave = zeros(all_ids.indmax)
        numinds = all_ids.sz
        start = time.time()
        for ids in all_ids:
            ids.copy()
            weave.inline('''
                for(int i=0; i<numinds; i++)
                    x_weave[ids[i]] += y[i];
                ''',
                ['x_weave', 'y', 'ids', 'numinds'],
                compiler='gcc',
                extra_compile_args=['-O3', '-march=native'],
                )
        weave_time = time.time()-start
        print 'Weave:', weave_time,
        if not do_loop:
            x_loop = x_weave
        print 'diff', amax(abs(x_loop-x_weave))#<1e-10

        def dobatchtiming(batchfunc, *args):
            global numops
            numops = 0
            x_batched = zeros(all_ids.indmax)
            def op(tgt, src):
                global numops
                numops += 1
                x_batched[tgt] += y[src]
            start = time.time()
            for ids in all_ids:
                batchfunc(op, ids.copy(), *args)
            batch_time = time.time()-start
            print batchfunc.__name__, ':', batch_time, '(%0.2f mean ops)'%((1.*numops)/all_ids.repeats),
            print 'diff', amax(abs(x_loop-x_batched)),
            print 'weavetime*%.1f'%(batch_time/weave_time)
            
        dobatchtiming(apply_batched_orig)
        dobatchtiming(apply_batched_orig_improved)
        dobatchtiming(apply_batched)
        dobatchtiming(apply_batched_take)
        dobatchtiming(apply_batched_take2)
