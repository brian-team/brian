#!/bin/bash

if [[ "$BRIAN_PACKAGE" == "" ]]; then
    BRIAN_PACKAGE=brian
fi

pylint --rcfile=dev/jenkins/pylint.rc $BRIAN_PACKAGE > pylint.log || :
