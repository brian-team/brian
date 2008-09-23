import os
os.chdir('../../../docs_sphinx') # work from docs_sphinx/ directory until we make the big change
# normally use build_html.py instead, faster
os.system('sphinx-build -a -E -b html . ../docs') # clean build