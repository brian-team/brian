import os, getpass, xmlrpclib, tempfile
from datetime import date
from matplotlib import pyplot as plt

from run_benchmarks import benchmarks, DB_PATH

TRAC_URL = 'https://%s:%s@neuralensemble.org/trac/brian/login/xmlrpc'
    
# Ask for username and password to authenticate at the TRAC server
username = raw_input('TRAC username [%s]: ' % getpass.getuser())
if not username:
    username = getpass.getuser()

password = getpass.getpass('TRAC password: ')


server = xmlrpclib.ServerProxy(TRAC_URL % (username, password))

print 'Getting the main wiki page.'
# get the benchmark start page:
start_page = server.wiki.getPage('benchmarks')

# assume there is a header: == List of benchmarks ==
header_index = start_page.find('== List of benchmarks ==')
if header_index == -1:
    raise ValueError('Wiki page does not contain "== List of benchmarks =="')

# leave everything before the header untouched
new_page = start_page[:header_index] + '== List of benchmarks ==\n' +\
           "''Last update:'' " + date.today().isoformat() + '\n'

for idx, benchmark in enumerate(benchmarks):    
    # add an entry with a link to the benchmark (the name for the benchmark    
    # wiki page is 'benchmark_' + its checksum)
    checksum, name = benchmark.checksum, benchmark.name
    wiki_name = 'benchmark_' + checksum
    
    print 'Adding/updating wiki page for Benchmark: ', name
    
    new_page += ' * [wiki:%s %s]\n' % (wiki_name, name)
    
    # Create the wikipage for the individual benchmark, including a link to the
    # graph
    benchmark_page = '''
= %(name)s =
''Description'': %(description)s

== Setup code ==
{{{
#!python

%(setup)s
}}}
== Benchmarked code ==
{{{
#!python

%(statement)s
}}}

== Benchmark results ==
[[Image(results.png, max-width:"100%%")]]
    ''' % {'name': benchmark.name,
           'description': benchmark.description,
           'setup': benchmark.setup,
           'statement': benchmark.code}
    
    try:
        old_page = server.wiki.getPage(wiki_name)
    except:
        old_page = None
        
    if old_page != benchmark_page:
        server.wiki.putPage(wiki_name, benchmark_page, {})
        
    # prepare the graph
    print '\tCreating graph'
    tmp_path = tempfile.mktemp(suffix='.png')   
    benchmark.plot(DB_PATH)        
    plt.savefig(tmp_path, dpi=150)
    
    print '\tUploading graph'
    # upload the graph as an attachment to the wiki page
    server.wiki.putAttachment('%s/results.png' % wiki_name,
                              xmlrpclib.Binary(open(tmp_path).read()))
    
    #delete the temporary file
    os.remove(tmp_path)        

# upload the main benchmark wiki page
if start_page != new_page:
    print 'Updating the main wiki page'
    server.wiki.putPage('benchmarks', new_page,
                        {'comment': 'benchmark update by %s' % username})
    
print 'All done'