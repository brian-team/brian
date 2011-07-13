from numpy import *
lines = open('brian_docs_analysis.txt', 'r').read().split('\n')
lines = [line for line in lines if '\t' in line]
header = lines[0].split('\t')
lines = [tuple(line.split('\t')) for line in lines[1:]]
logdtype = [('page', object),
            ('visits', int),
            ('views', int),
            ('duration', float),
            ('sorties', float),
            ('value', float)]
print header
data = array(lines, dtype=logdtype)
data.sort(order='visits')
data = data[::-1]

def write_data(data, fname, n=100):
    with open(fname, 'w') as f:
        f.write('\t'.join(map(str, header))+'\n')
        for d in data[:n]:
            f.write('\t'.join(map(str, d))+'\n')

docs_data = data[array([d.startswith('/docs/') for d in data['page']])]
write_data(docs_data, 'topdocs_visits.txt')

docs_data_html = docs_data[array([d.endswith('.html') for d in docs_data['page']])]
write_data(docs_data_html, 'topdocs_visits_html.txt')

I = argsort(-docs_data_html['views']*1.0/docs_data_html['visits'])
docs_data_html_revisited = docs_data_html[I]
docs_data_html_revisited = docs_data_html_revisited[array([d>20 for d in docs_data_html_revisited['views']])]
write_data(docs_data_html_revisited, 'topdocs_revisited_html.txt')

I = argsort(-docs_data_html['duration'])
docs_data_html_duration = docs_data_html[I]
docs_data_html_duration = docs_data_html_duration[array([d>20 for d in docs_data_html_duration['views']])]
write_data(docs_data_html_duration, 'topdocs_duration_html.txt')
