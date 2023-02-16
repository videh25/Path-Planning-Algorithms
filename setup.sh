#! /usr/bin/bash

# Adds the direcotry path to python path, for consistent referencing
if [ -z "$PYTHONPATH" ]
then
    export PYTHONPATH="$(pwd)"
else
    export PYTHONPATH="$PYTHONPATH:$(pwd)"
fi