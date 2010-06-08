import os, glob
curdir = os.getcwd()
#print "Generating interface..."
#os.system('generate_interface.py')
#os.chdir(curdir)
print "Generating tutorials..."
os.system('generate_tutorials.py')
os.chdir(curdir)
print "Generating preference list..."
os.system('generate_prefsdocs.py')
os.chdir(curdir)
print "Generating examples..."
os.system('generate_examples.py')
os.chdir(curdir)
print "Generating HTML..."
# use clean build because this script will typically be called for new releases only
import build_html_clean
print "Done"
