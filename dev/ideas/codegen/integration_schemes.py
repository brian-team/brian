__all__ = ['euler_scheme', 'rk2_scheme', 'exp_euler_scheme']

euler_scheme = [
    (('foreachvar', 'all'),
        '''
        $vartype ${var}__tmp = $var_expr
        '''),
    (('foreachvar', 'all'),
        '''
        $var += ${var}__tmp*dt
        ''')
    ]

rk2_scheme = [
    (('foreachvar', 'all'),
        '''
        $vartype ${var}__buf = $var_expr
        $vartype ${var}__half = (.5*dt)*${var}__buf
        ${var}__half += $var
        '''),
    (('foreachvar', 'all'),
        '''
        ${var}__buf = @substitute(var_expr, dict((var, var+'__half') for var in vars))
        $var += dt*${var}__buf
        ''')
    ]

exp_euler_scheme = [
    (('foreachvar', 'all'),
        '''
        $vartype ${var}__B = @substitute(var_expr, {var:0})
        $vartype ${var}__A = @substitute(var_expr, {var:1})
        ${var}__A -= ${var}__B
        ${var}__B /= ${var}__A
        ${var}__A *= dt
        '''),
    (('foreachvar', 'all'),
        '''
        $var += ${var}__B
        $var *= exp(${var}__A)
        $var -= ${var}__B
        ''')
    ] 
