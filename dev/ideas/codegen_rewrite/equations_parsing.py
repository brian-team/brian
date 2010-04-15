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
    lines = eqs.split('\n')
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line]
    lines = map(sides, lines)
    replacements = {}
    for lhs, rhs in lines:
        lhs = replace_differentials(lhs, replacements)
        rhs = replace_differentials(rhs, replacements)
        print lhs, '=', rhs
    for k, (v, n, r) in replacements.iteritems():
        print 'Symbol', k, 'defines order', n, 'differential of', v, ':', r

if __name__=='__main__':
    eqs = '''
    a * d2x/dt2 + b * d2y/dt2 = c
    d * d2x/dt2 + e * d2y/dt2 = f
    '''
    parse(eqs)
