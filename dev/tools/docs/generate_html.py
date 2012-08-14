import os, glob

pathname = os.path.abspath(os.path.dirname(__file__))
os.chdir(pathname)

# Tutorials, preferences and examples only for brian (not brian 2)
if not ('DOCNAME' in os.environ) or os.environ['DOCNAME'] == 'brian':
    print "Generating tutorials..."
    import generate_tutorials
    
    os.chdir(pathname)
    print "Generating preference list..."
    import generate_prefsdocs    
    
    os.chdir(pathname)
    print "Generating examples..."
    import generate_examples

os.chdir(pathname)
print "Generating HTML..."
# use clean build because this script will typically be called for new releases only
import build_html_clean
print "Done"
