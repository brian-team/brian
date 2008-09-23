import os

# delete this when make_new_release.py is working
exclude = [ 'autodiff' ]

def api_command_line(exclude):
    s = 'epydoc.py -o docs/api --docformat plaintext --no-private --no-frames --no-sourcecode '
    if len(exclude):
        s += '--exclude="'+'|'.join(exclude)+'" '
    s += 'brian'
    return s 

if __name__=='__main__':
    basepathname, filename = os.path.split(__file__)
    os.chdir(basepathname)
    os.chdir('../../../.') # work from root directory
    os.system(api_command_line(exclude))