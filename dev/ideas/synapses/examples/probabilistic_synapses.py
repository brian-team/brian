"""
Probabilistic synapses
"""
S=Synapses(source,target,model="""w : 1
                                  p : 1 # transmission probability""",
                         pre="v+=w*(rand(n)<p)")
