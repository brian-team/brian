'''
Brian screensaver

Uses pyscr (google "Python screensaver")
'''
import pyscr


class MySaver(pyscr.Screensaver):
    #set up timer for tick() calls
    TIMEBASE=1. # called every second

    def configure(self):
        #called to open the screensaver configuration
        #maybe implement here something with venster or win32all
        from ctypes import windll
        windll.user32.MessageBoxA(0,
            "This is the configuration of the Brian screensaver.",
            "Brian Screensaver Configuration", 0, 0, 0)

    def initialize(self):
        #called once when the screensaver is started
        pass

    def finalize(self):
        #called when the screensaver terminates
        pass

    def tick(self):
        #called when the timer tick occurs, set up with startTimer from above
        self.dc.beginDrawing()
        self.dc.setColor(0xffffffL)
        self.dc.setTextColor(0xff0000L)
        self.dc.setBgColor(0x000000L)
        self.dc.setFont("arial")
        #~ self.dc.setBgTransparent(True)
        self.dc.drawText((100, 100), "Brian is thinking...")
        w, h=self.dc.getSize()
        self.dc.drawRect((0, 0), (w-1, h-1))
        self.dc.fillEllipse((50, 50), (60, 60))
        self.dc.endDrawing()

#standard 'main' detection and startof screensaver
if __name__=='__main__':
    pyscr.main()
