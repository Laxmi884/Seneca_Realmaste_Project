#!/bin/bash

# set config file
SCRIPT=$(readlink -f "$0")
SCRIPTPATH=$(dirname "$SCRIPT")
export RMBASE_FILE_PYTHON=$SCRIPTPATH/cfg.py

export DEBUG=True

# active python virtualenv if needed
# source $HOME/py3/bin/activate

# run python script
python $SCRIPTPATH/organizer.py $*
