#!/bin/bash
echo "Starting Flask API..."
export FLASK_APP=app/app.py
flask run --host=0.0.0.0 --port=5000
