def bis(x, min, max, i=None, j=None):
    if i is None:
        i = 0
    if j is None:
        j = len(x)
    L = i
    R = j-1
    while L<R-1:
        print (L, R)
        if x[L]>=min:
            break
        M = int((L+R)/2)
        if x[M]<min:
            L = M;
        else:
            R = M;
    if x[L]<min:
        L = R
    print '1:', (L, R)
    i0 = L
    R = j-1
    while L<R-1:
        print (L, R)
        if x[L]>=max:
            break
        M = int((L+R)/2)
        if x[M]<max:
            L = M;
        else:
            R = M;
    if x[L]<max:
        L = R
    print '2:', (L, R)
    j0 = L
    return x[i0:j0]

def bisect_left(a, x, lo=0, hi=None):
    """Return the index where to insert item x in list a, assuming a is sorted.

    The return value i is such that all e in a[:i] have e < x, and all e in
    a[i:] have e >= x.  So if x already appears in the list, i points just
    before the leftmost x already there.

    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    """

    if hi is None:
        hi = len(a)
    while lo < hi:
        mid = (lo+hi)//2
        if a[mid] < x: lo = mid+1
        else: hi = mid
    return lo

def bis(x, min, max, i=0, j=None):
    if j is None:
        j = len(x)
    lo = i
    hi = j
    while lo<hi:
        mid = (lo+hi)//2
        if x[mid]<min: lo = mid+1
        else: hi = mid
    i0 = lo
    hi = j
    while lo<hi:
        mid = (lo+hi)//2
        if x[mid]<max: lo = mid+1
        else: hi = mid
    j0 = lo
    return x[i0:j0]

print bis([30,30,31,31,31,35,37,37,37,38,38,50],
          31,38)
    