#!/usr/bin/env bash
export FLASK_APP=run.py
gunicorn -w 1 -b 0.0.0.0:5000 run:app