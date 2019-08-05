#!/bin/bash
cd "$HOME/apps/unredactor/unredactor"
source .venv/bin/activate
uwsgi --socket 0.0.0.0:8000 --protocol=http -w wsgi:application

