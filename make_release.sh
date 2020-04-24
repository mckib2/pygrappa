#!/usr/bin/env bash

## TO RUN:
##     source path/to/venv/bin/activate
##     source make_release.sh

# Remove any existing distribution archives
rm -rf dist
mkdir dist

# Make sure we have the latest Cython
python -m pip install --upgrade Cython

# Generate distribution archives
python -m pip install --upgrade setuptools wheel
python setup.py sdist # bdist_wheel

# Upload
python -m pip install --upgrade twine
python -m twine upload dist/*
