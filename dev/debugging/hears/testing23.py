from brian import *
from brian.hears import *
    
class RectifiedGammatone(CombinedFilterbank):
    def __init__(self, source, cf):
        CombinedFilterbank.__init__(self, source)
        source = self.get_modified_source()
        gfb = Gammatone(source, cf)
        output = FunctionFilterbank(gfb, lambda input: clip(input, 0, Inf))
        self.set_output(output)
        
x = whitenoise(100*ms)
fb = RectifiedGammatone(x, [1*kHz, 1.5*kHz])
y = fb.process()
subplot(211)
plot(y)
subplot(212)
fb.source = tone(1*kHz, 100*ms)
y = fb.process()
plot(y)
show()
