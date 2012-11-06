#!/bin/bash

if [[ "$BRIAN_PACKAGE" == "" ]]; then
    BRIAN_PACKAGE=brian
fi

find $BRIAN_PACKAGE -not -path "$BRIAN_PACKAGE/tests/*" -name '*.py' -not -name '__init__.py' -print0 | xargs -0 pyflakes > pyflakes_warnings.log 2> pyflakes_errors.log || :
