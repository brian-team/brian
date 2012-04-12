# Use previously built Brian 
export PYTHONPATH=$(pwd)/build/lib:$PYTHONPATH
cd dev/tests/
rm examples_completed.txt || :

# get the newest version of nose
BASEDIR=$(dirname $PYTHON_EXE)
$BASEDIR/pip install --upgrade -I nose || :

# Do not fail if one of the examples fails, instead the run will be marked as
# "unstable" by the violations plugin
python -u allexamples_cumulative.py || :
