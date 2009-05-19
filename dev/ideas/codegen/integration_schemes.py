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
        $vartype ${var}__half = $var+dt*${var}__buf
        '''),
    (('foreachvar', 'all'),
        '''
        ${var}__buf = @substitute(var_expr, {var:var+'__buf'})
        $var += dt*${var}__buf
        ''')
    ]

exp_euler_scheme = [
    (('foreachvar', 'all'),
        '''
        $vartype ${var}__B = @substitute(var_expr, {var:0})
        $vartype ${var}__A = @substitute(var_expr, {var:1})-${var}__B
        ${var}__B /= ${var}__A
        '''),
    (('foreachvar', 'all'),
        '''
        $var = ($var+${var}__B)*exp(${var}__A*dt)-${var}__B
        ''')
    ] 
