euler_scheme = '''
intermediate:
    $vartype ${var}__tmp = $var_expr
update:
    $var += ${var}__tmp
'''

rk2_scheme = '''
intermediate:
    $vartype ${var}__buf = @substitute($var_expr, '$var')
    $vartype ${var}__half = $var+dt*${var}__buf
update:
    ${var}__buf = @substitute($var_expr, '$var__half')
    $var += dt*${var}__buf
'''

exp_euler_scheme = '''
intermediate:
    $vartype ${var}__B = @value_substitute($var_expr, '$var', $var, 0)
    $vartype ${var}__A = @value_substitute($var_expr, '$var', $var, 1)-${var}__B
    ${var}__B /= ${var}__A
update:
    $var = ($var+${var}__B)*exp(${var}__A*dt)-${var}__B 
'''

###

euler_scheme = '''
@foreachneuron
    @foreachvar: $all_vars
        $vartype ${var}__tmp = $var_expr
    @foreachvar: $all_vars
        $var += ${var}__tmp
'''

rk2_scheme = '''
@foreachneuron
    @foreachvar: $all_vars
        $vartype ${var}__buf = @substitute($var_expr, '$var')
        $vartype ${var}__half = $var+dt*${var}__buf
    @foreachvar: $all_vars
        ${var}__buf = @substitute($var_expr, '$var__half')
        $var += ${var}__tmp
'''

exp_euler_scheme = '''
@foreachneuron
    @foreachvar: $all_vars
        $vartype ${var}__B = @value_substitute($var_expr, '$var', $var, 0)
        $vartype ${var}__A = @value_substitute($var_expr, '$var', $var, 1)-${var}__B
        ${var}__B /= ${var}__A
    @foreachvar: $all_vars
        $var = ($var+${var}__B)*exp(${var}__A*dt)-${var}__B 
'''