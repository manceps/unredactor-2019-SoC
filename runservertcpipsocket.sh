#!/bin/bash
source activate.sh
uwsgi --socket 0.0.0.0:8000 --protocol=http -w wsgi:application

