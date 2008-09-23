import os, re, glob, brian, inspect, compiler, unicodedata
documentable_names = set()
for k, v in brian.__dict__.iteritems():
    try:
        if 'brian' in inspect.getsourcefile(v):
            documentable_names.add(k)
    except TypeError:
        pass
os.chdir('../../../examples')
examplesfnames = [fname for fname in glob.glob('*.py') if 'parallelpythonised' not in fname]
examplesbasenames = [fname[:fname.find('.')] for fname in examplesfnames]
examplescode = [open(fname,'r').read() for fname in examplesfnames]
examplesdocs = []
examplesafterdoccode = []
examplesdocumentablenames = []
for code in examplescode:
    codesplit = code.split('\n')
    readingdoc = False
    doc = []
    afterdoccode = ''
    for i in range(len(codesplit)):
        stripped = codesplit[i].strip()
        if stripped[:3]=='"""' or stripped[:3]=="'''":
            if not readingdoc:
                readingdoc = True
            else:
                afterdoccode = '\n'.join(codesplit[i+1:])
                break
        elif readingdoc:
            doc.append(codesplit[i])
        elif not stripped or stripped[0]=='#':
            pass
        else:
            break
    doc = '\n'.join(doc)
    # next line replaces unicode characters like e-acute with standard ascii representation
    examplesdocs.append(unicodedata.normalize('NFKD',unicode(doc,'latin-1')).encode('ascii','ignore'))
    examplesafterdoccode.append(afterdoccode)
    examplesdocumentablenames.append(set(compiler.compile(code,'','exec').co_names) & documentable_names)
examples = zip(examplesfnames, examplesbasenames, examplescode, examplesdocs, examplesafterdoccode, examplesdocumentablenames)
os.chdir('../docs_sphinx')
for fname, basename, code, docs, afterdoccode, documentables in examples:
    title = 'Example: '+basename
    output = '.. currentmodule:: brian\n\n'
    output += '.. _example-'+basename+':\n\n'
    if len(documentables):
        output += '.. index::\n'
        for dname in documentables:
            output += '   pair: example usage; '+dname+'\n'
        output += '\n'
    output += title+'\n'+'='*len(title)+'\n\n'
    output += docs + '\n\n::\n\n'
    output += '\n'.join(['    '+line for line in afterdoccode.split('\n')])
    output += '\n\n'
    open('examples-'+basename+'.txt','w').write(output)

mainpage_text = open('examples.txt','r').read()
mainpage_text = mainpage_text[:mainpage_text.find('.. EXAMPLES TOCTREE')]
mainpage_text += '.. EXAMPLES TOCTREE\n\n'
mainpage_text += '.. toctree::\n'
mainpage_text += '   :maxdepth: 1\n\n'
for fname, basename, code, docs, afterdoccode, documentables in examples:
    mainpage_text += '   examples-'+basename+'\n'
open('examples.txt','w').write(mainpage_text)