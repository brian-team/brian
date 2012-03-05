find brian -name '*.py' | egrep -v '/tests/' | xargs pyflakes > pyflakes_warnings.log 2> pyflakes_errors.log || :
