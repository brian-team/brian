import os, sys, brian, re, datetime

def setreleasedate():
    releasedate = str(datetime.date.today())
    pathname = os.path.abspath(os.path.dirname(__file__))
    os.chdir(pathname)
    os.chdir('../../../')
    # update __init__.py
    init_py = open('brian/__init__.py', 'r').read()
    init_py = re.sub("__release_date__\s*=\s*'.*?'", "__release_date__ = '" + releasedate + "'", init_py)
    open('brian/__init__.py', 'w').write(init_py)

if __name__ == '__main__':
    setreleasedate()
