import os, sys, getpass, xmlrpclib, tempfile
from datetime import date, datetime
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.dates import DateFormatter, MonthLocator

from run_benchmarks import benchmarks

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print 'Need the database filename as an argument'
        sys.exit(1)
    
    DB_PATH = sys.argv[1]

    # BRIAN releases (some minor left out that occurred in too close succession)
    releases = {datetime(2008, 9, 23) : '1.0',
                datetime(2009, 12, 8) : '1.1.3',
                datetime(2010, 1, 19) : '1.2.0',
                datetime(2010, 9, 27) : '1.2.1',
                datetime(2011, 2, 18) : '1.3.0',
                datetime(2011, 12, 22) : '1.3.1',
                datetime(2012, 8, 29) : '1.4.0'}
    
    TRAC_URL = 'https://%s:%s@neuralensemble.org/trac/brian/login/xmlrpc'
    
    # Ask for username and password to authenticate at the TRAC server
    
    # First look for environment variables (set in the Jenkins job)
    username = os.getenv('TRAC_USERNAME', None)
    password = os.getenv('TRAC_PASSWORD', None)
    
    if username is None:
        username = raw_input('TRAC username [%s]: ' % getpass.getuser())
        if not username:
            username = getpass.getuser()
    
    if password is None:
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
    new_page = start_page[:header_index] + '== List of benchmarks ==\n' + \
               "''Last update:'' " + date.today().isoformat() + '\n'
    
    # sort benchmarks according to name
    benchmarks.sort(key=lambda b: b.name)
    
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
            server.wiki.putPage(wiki_name, benchmark_page, {'minoredit' : True})
    
        # prepare the graph
        print '\tCreating graph'
        tmp_path = tempfile.mktemp(suffix='.png')
        results = benchmark.get_results(DB_PATH)['timing']
        
        # make basic plot with grid in background
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_axis_bgcolor('#eeeeec')
        if results.max() > 2500:
            divider = 1000.
            unit = 's'
        else:
            divider = 1.
            unit = 'ms'
        ax.plot(results.index, results / divider, color='#2e3436', linewidth=1.5)
        max_y = ax.get_ylim()[1]
        ax.set_ylim(0, max_y)
        ax.grid(True, color='white', linestyle='-', linewidth=3)
        ax.set_axisbelow(True)
        
        # Disable spines.
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        
        # Disable ticks.
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        
        # format dates
        formatter = DateFormatter("%m/%Y")
        ax.xaxis.set_major_locator(MonthLocator(bymonth=(1, 4, 7, 10)))
        ax.xaxis.set_major_formatter(formatter)
        fig.autofmt_xdate(rotation=45)
        
        ax.set_title(benchmark.name)
        ax.set_xlabel('Date')
        ax.set_ylabel('Run time (%s)' % unit)    
        
        # add information about releases
        for date, release in releases.iteritems():
            ax.plot([date, date], [0, max_y], color='#ef2929')    
            ax.annotate(release, (date, 0.025 * max_y), color='#a40000', 
                        verticalalignment='left', rotation=45)
        plt.tight_layout()
    
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
                            {'comment': 'benchmark update by %s' % username,
                            'minoredit' : True})
    
    print 'All done'
