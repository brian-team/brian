from brian import *
#set_global_preferences(usecstdp=False)
#log_level_debug()

G = NeuronGroup(1, 'V:1', reset=0, threshold=1)
C = Connection(G, G, 'V',
               delay=True, max_delay=5*ms
               )

eqs_stdp = """
dA_pre/dt  = -A_pre/(10*ms)   : 1
dA_post/dt = -A_post/(10*ms) : 1
"""
stdp = STDP(C, eqs=eqs_stdp,
            pre='''
            A_pre += 1
            w += A_pre*A_post
            A_pre += 1
            ''',
            post='''
            A_post += 1
            w += A_pre
            ''', wmax=1)

G.V = 2

run(1*ms)
