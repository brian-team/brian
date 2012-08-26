#!/bin/bash

if [[ "$BRIAN_PACKAGE" == "" ]]; then
    BRIAN_PACKAGE=brian
fi

find $BRIAN_PACAKGE -name '*.py' | egrep -v '/tests/' | xargs pyflakes > pyflakes_warnings.log 2> pyflakes_errors.log || :
