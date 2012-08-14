import os

if 'DOCROOT' in os.environ:
    os.chdir(os.environ['DOCROOT'])
else:
    os.chdir('../../../')
    pathname = os.path.abspath(os.path.dirname(__file__))
    os.chdir(pathname)
    os.chdir('../../../') # work from docs_sphinx/ directory until we make the big change

os.chdir('docs_sphinx')
# if this doesn't work, run build_html_clean.py instead
os.system('sphinx-build -a -E -D building_as=latex -b latex . ./_latexbuild')
os.chdir('_latexbuild')
os.system('make')
