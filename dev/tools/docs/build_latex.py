import os

pathname = os.path.abspath(os.path.dirname(__file__))
os.chdir(pathname)

os.chdir('../../../docs_sphinx') # work from docs_sphinx/ directory until we make the big change
# if this doesn't work, run build_html_clean.py instead
os.system('sphinx-build -a -E -D building_as=latex -b latex . ./_latexbuild')
os.chdir('_latexbuild')
os.system('make')
