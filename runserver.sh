#!/bin/bash
# source activate.sh
# uwsgi --socket 0.0.0.0:8000 --protocol=http -w wsgi:application

UNREDACTOR_DIR="`dirname \"$0\"`"              # relative
UNREDACTOR_DIR="`( cd \"$UNREDACTOR_DIR\" && pwd )`"  # absolutized and normalized
if [ -z "$UNREDACTOR_DIR" ] ; then
  # error; for some reason, the path is not accessible
  # to the script (e.g. permissions re-evaled after suid)
  exit 1  # fail
fi
export UNREDACTOR_DIR=$(dirname $UNREDACTOR_DIR)/unredactor/unredactor
echo "UNREDACTOR_DIR=$UNREDACTOR_DIR"

cd $UNREDACTOR_DIR
source $UNREDACTOR_DIR/.venv/bin/activate
uwsgi --ini "$UNREDACTOR_DIR/unredactor.ini"
