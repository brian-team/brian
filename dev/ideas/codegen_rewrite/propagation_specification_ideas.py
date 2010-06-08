from brian.inspection import namespace
import re


class Symbol(str):
    pass


class OutputCode(object):
    def __init__(self, source=None):
        self.code = ''
        if source is None:
            source = self
        self.source = source
        self.curtarget = self

    def newtarget(self):
        self.curtarget = OutputCode(self.curtarget)
        return self.curtarget

    def close(self):
        self.source.code = self.source.code + self.curtarget.code
        self.curtarget = self.curtarget.source

    def add(self, line):
        self.curtarget.code = self.curtarget.code + line

    def __str__(self):
        s = ''
        tabs = 0
        for line in self.code.split('\n'):
            if '}' in line and '{' not in line: tabs -= 4
            s += ' ' * tabs + line + '\n'
            if '{' in line and '}' not in line: tabs += 4
        return s
    __repr__ = __str__


class Maker(object):
    def __init__(self, cls, outputcode):
        self.outputcode = outputcode
        self.cls = cls
        self.instance = 0

    def __iter__(self):
        self.instance += 1
        return self.cls(self.outputcode.newtarget(), self.instance)


class Spikes(object):
    def __init__(self, targetcode, instance=0):
        self.instance = instance
        self.targetcode = targetcode
        self.targetcode.add('for(int spike_index=0; spike_index<spikes_len; spike_index++){\n')
        self.targetcode.add('int neuron_index%d = spikes[spike_index];\n' % instance)
        self.value = 0

    def __iter__(self):
        return self

    def next(self):
        if self.value == 0:
            self.value = 1
            return Symbol('neuron_index' + str(self.instance))
        self.targetcode.add('}\n')
        self.targetcode.close()
        raise StopIteration


class Evaluate(object):
    def __init__(self, outputcode):
        self.outputcode = outputcode

    def __call__(self, code):
        ns = namespace(code, level=1)
        for k, v in ns.iteritems():
            if isinstance(v, Symbol):
                code = re.sub("\\b" + k + "\\b", str(v), code)
        self.outputcode.add(''.join(expr + ';\n' for expr in code.split('\n') if expr.strip()))


class DenseMatrix(object):
    def __init__(self, outputcode, matrix_name):
        self.outputcode = outputcode
        self.matrix_name = matrix_name
        self.instance = 0

    def row(self, row):
        self.instance += 1
        return DenseMatrixRow(row, self.matrix_name,
                              self.outputcode.newtarget(), self.instance)


class DenseMatrixRow(object):
    def __init__(self, row, matrix_name, targetcode, instance):
        self.targetcode = targetcode
        self.instance = instance
        self.row = row
        self.matrix_name = matrix_name
        self.targetcode.add('for(int col_index%d=0; col_index%d<%s_num_cols; col_index%d++){\n' % (instance, instance, matrix_name, instance))
        self.targetcode.add('double &matrix_value%d = %s[col_index%d+%s*%s_num_cols];\n' % (instance, matrix_name, instance, row, matrix_name))
        self.value = 0

    def __iter__(self):
        return self

    def next(self):
        if self.value == 0:
            self.value = 1
            return Symbol('col_index' + str(self.instance)), Symbol('matrix_value' + str(self.instance))
        self.targetcode.add('}\n')
        self.targetcode.close()
        raise StopIteration


class Load(object):
    def __init__(self, outputcode):
        self.outputcode = outputcode

    def __call__(self, code):
        addcode = ''
        for line in code.split('\n'):
            for expr in line.split(';'):
                if expr.strip():
                    m = re.search('(\w*)\[(.*?)\]', expr)
                    if not m:
                        raise ValueError('Load string incorrent format: ' + code)
                    varname = m.group(1)
                    index = m.group(2)
                    addcode += '%s = &%s__array[%s];\n' % (varname, varname, index)
        self.outputcode.add(addcode)

if __name__ == '__main__':
    outputcode = OutputCode()
    spikes = Maker(Spikes, outputcode)
    evaluate = Evaluate(outputcode)
    load = Load(outputcode)
    W = DenseMatrix(outputcode, 'W')

    for i in spikes:
        for j, w in W.row(i):
            load('V[i]')
            evaluate('w += V')

    print outputcode

    # This works as follows, when you do for example:
    #    for i in spikes:
    # the object spikes knows that it should output code to the OutputCode object
    # outputcode. The spikes object has a method __iter__ and this returns a
    # Spikes object s, which follows the iteration protocol so that s.next()
    # returns either the next object in the container or raises StopIteration.
    # In fact, we only use it once, the first time we add some code to outputcode
    # to do the iteration, and return a Symbol object which is a string with
    # the name of the neuron index being iterated over. So for example it adds
    # the code:
    #    for(int spike_index=0; spike_index<spikes_len; spike_index++){
    #        int neuron_index1 = spikes[spike_index];
    # and returns Symbol('neuron_index1') so that i=Symbol('neuron_index1').
    # Now if other code uses 'i' or i then it will be replaced by neuron_index1.
    # The second time the iterator is called it adds the code '}' to outputcode
    # and then raises StopIteration. The effect of this is that the for block
    # is executed precisely once, wrapped inside a C for-loop.
    # The other examples like W.row(i), load('V[i]') and evaluate('w += V') work
    # in a similar way.
    #
    # Questions:
    # * can we generate Python code with the same system?
    # * can we structure the code above to make it simpler? e.g. putting C
    #   and Python code into template files rather than having them mixed
    #   together?
