'''
Apply style changes to all Brian modules.

From PEP-8 (Python naming conventions):
---------------------------------------
Package and Module Names

      Modules should have short, all-lowercase names.  Underscores can be used
      in the module name if it improves readability.  Python packages should
      also have short, all-lowercase names, although the use of underscores is
      discouraged.

      Since module names are mapped to file names, and some file systems are
      case insensitive and truncate long names, it is important that module
      names be chosen to be fairly short -- this won't be a problem on Unix,
      but it may be a problem when the code is transported to older Mac or
      Windows versions, or DOS.

      When an extension module written in C or C++ has an accompanying Python
      module that provides a higher level (e.g. more object oriented)
      interface, the C/C++ module has a leading underscore (e.g. _socket).

    Class Names

      Almost without exception, class names use the CapWords convention.
      Classes for internal use have a leading underscore in addition.

    Exception Names

      Because exceptions should be classes, the class naming convention
      applies here.  However, you should use the suffix "Error" on your
      exception names (if the exception actually is an error).

    Global Variable Names

      (Let's hope that these variables are meant for use inside one module
      only.)  The conventions are about the same as those for functions.

      Modules that are designed for use via "from M import *" should use the
      __all__ mechanism to prevent exporting globals, or use the older
      convention of prefixing such globals with an underscore (which you might
      want to do to indicate these globals are "module non-public").

    Function Names

      Function names should be lowercase, with words separated by underscores
      as necessary to improve readability.

      mixed_case is allowed only in contexts where that's already the
      prevailing style (e.g. threading.py), to retain backwards compatibility.

    Function and method arguments

      Always use 'self' for the first argument to instance methods.

      Always use 'cls' for the first argument to class methods.

      If a function argument's name clashes with a reserved keyword, it is
      generally better to append a single trailing underscore rather than use
      an abbreviation or spelling corruption.  Thus "print_" is better than
      "prnt".  (Perhaps better is to avoid such clashes by using a synonym.)

    Method Names and Instance Variables

      Use the function naming rules: lowercase with words separated by
      underscores as necessary to improve readability.

      Use one leading underscore only for non-public methods and instance
      variables.

      To avoid name clashes with subclasses, use two leading underscores to
      invoke Python's name mangling rules.

      Python mangles these names with the class name: if class Foo has an
      attribute named __a, it cannot be accessed by Foo.__a.  (An insistent
      user could still gain access by calling Foo._Foo__a.)  Generally, double
      leading underscores should be used only to avoid name conflicts with
      attributes in classes designed to be subclassed.

      Note: there is some controversy about the use of __names (see below).
'''

import os, re

os.chdir('..')
os.chdir('..')
os.chdir('..') # work from Brian's root

#package_names={'Brian':'brian'}
#module_names={'BrianNoUnits':'brian_no_units'}
specific_changes={'is_number_type':'isNumberType',
                  'is_sequence_type':'isSequenceType',
                  'assert_raises':'assertRaises',
                  'BrianNoUnits':'brian_no_units',
                  'BrianNoUnitsNoWarnings':'brian_no_units_no_warnings',
                  'BrianUnitPrefs':'brian_unit_prefs',
                  'from brian':'from brian',
                  'Library':'library'}

def bad_identifiers(text):
    return set(re.findall(r'\b[a-z0-9_]+[A-Z]\w+\b', text))

def replace_bad_identifiers(text, names):
    text2=text
    for name in names:
        new_name=re.sub('[A-Z]', lambda c:'_'+c.group(0).lower(), name)
        text2=re.sub(name, new_name, text2)
    return text2

def make_specific_changes(text):
    text2=text
    for name1, name2 in specific_changes.iteritems():
        text2=re.sub(name1, name2, text2)
    return text2

def patch_file(filename):
    f_in=open(filename)
    text=f_in.read()
    f_in.close()

    names=bad_identifiers(text)
    text_out=text
    if names!=set([]):
        text_out=replace_bad_identifiers(text_out, names)
    text_out=make_specific_changes(text_out)

    if text!=text_out:
        f_out=open(filename, 'w')
        f_out.write(text_out)
        f_out.close()

def patch_all_files(verbose=True):
    # Get all source files
    files=[]
    for dirpath, dirnames, filenames in os.walk('.'):
        files.extend([dirpath+'\\'+file for file in filenames if file[-3:]=='.py'])
    for file in files:
        if file[-13:]!='stylepatch.py':
            if verbose:
                print file
            patch_file(file)

if __name__=='__main__':
    patch_all_files()
