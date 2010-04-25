'''
TODO: for some equations should we first reduce and then solve rather than
      solving and then reducing? probably...
'''

from sympy import Symbol, solve
import re

def sides(line):
    m = re.match(r'([^=]*)=([^e]*)', line)
    lhs, rhs = m.groups()
    return lhs.strip(), rhs.strip()

def replace_differentials(expr, replacements):
    while True:
        m = re.search(r'd\s*(\d*)\s*(\w+)\s*/\s*d\s*t\s*(\d*)', expr)
        if not m: break
        n1, var, n2 = m.groups()
        if n1!=n2:
            raise ValueError('Inconsistent indices in differential')
        if n1=='':
            n1 = '1'
        n = int(n1)
        newsym = '_diff_'+var+'_'+n1
        replacements[newsym] = (var, n, expr[m.start():m.end()])
        expr = expr[:m.start()] + newsym + expr[m.end():]
    return expr

def parse(eqs):
    print 'Original equations:'
    print '\n'.join(line.strip() for line in eqs.strip().split('\n'))
    
    lines = eqs.split('\n')
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line]
    lines = map(sides, lines)
    replacements = {}
    eqstosolve = []
    print
    print 'Equations after symbol replacement:'
    for lhs, rhs in lines:
        lhs = replace_differentials(lhs, replacements)
        rhs = replace_differentials(rhs, replacements)
        eqstosolve.append('(%s)-(%s)' % (lhs, rhs))
        print lhs, '=', rhs
    for k, (v, n, r) in replacements.iteritems():
        print 'Symbol', k, 'defines order', n, 'differential of', v, ':', r
    # Now solve this system of equations for the differentials
    symbolset = dict((k, Symbol(k)) for k in replacements.keys())
    ns = symbolset.copy()
    ns.update(dict((v, Symbol(v)) for v, _, _ in replacements.values()))
    eqstosolve = [eval(eq, ns) for eq in eqstosolve]
    solutions = solve(eqstosolve, *symbolset.values())
    print
    print 'Solutions:'
    for k, v in solutions.items():
        print k, '=', v
    # Now reduce Nth order equations to  first order ones
    reducedeqs = {}
    print
    print 'Reduced set of equations:'
    for diffeqsymbol, diffeqvalue in solutions.items():
        var, n, r = replacements[str(diffeqsymbol)]
        for i in xrange(n):
            if i==0:
                newvar = var
            else:
                newvar = '_rdiff_%s_%d'%(var, i)
            if i==n-1:
                neweq = diffeqvalue
            else:
                neweq = '_rdiff_%s_%d'%(var, i+1)
            reducedeqs[newvar] = neweq
            print 'd%s/dt = %s'%(newvar, neweq) 

if __name__=='__main__':
#    eqs = '''
#    a * d2x/dt2 + b * d2y/dt2 = c
#    d * d2x/dt2 + e * d2y/dt2 = f
#    '''
    eqs = '''
    3 * d2x/dt2 + 2 * d2y/dt2 = 6*x
    4 * d2x/dt2 + 1 * d2y/dt2 = 5*y
    '''
    parse(eqs)
