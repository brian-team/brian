#!/bin/bash

if [[ "$BRIAN_PACKAGE" == "" ]]; then
    BRIAN_PACKAGE=brian
fi

clonedigger --cpd-output --ignore-dir=tests --ignore-dir=experimental -o clonedigger.xml $BRIAN_PACKAGE || :
