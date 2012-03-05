# Use previously built Brian 
export PYTHONPATH=$(pwd)/build/lib:$PYTHONPATH
cd dev/tests/
rm examples_completed.txt || :

# Do not fail if one of the examples fails, instead the run will be marked as "unstable" by the text-finder plugin
python -u allexamples_cumulative.py || :
