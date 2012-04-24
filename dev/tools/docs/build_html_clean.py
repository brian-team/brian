import os, shutil, sys
# remove old docs
os.chdir('../../../')
if os.path.exists('docs'):
    shutil.rmtree('docs')
os.mkdir('docs')
# Generate new docs
os.chdir('docs_sphinx') # work from docs_sphinx/ directory until we make the big change
# normally use build_html.py instead, faster
sys.exit(os.system('sphinx-build -W -a -E -b html . ../docs')) # clean build
