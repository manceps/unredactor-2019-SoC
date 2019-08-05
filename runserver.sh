#!/bin/bash
export UNREDACTOR_DIR="$HOME/apps/unredactor/unredactor"
cd $UNREDACTOR_DIR
source .venv/bin/activate
# uwsgi --socket 0.0.0.0:8000 --protocol=http -w wsgi:application
uwsgi --ini "$UNREDACTOR_DIR/unredactor.ini"
