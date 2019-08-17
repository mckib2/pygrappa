#!/usr/bin/env bash

## TO RUN:
##     source path/to/venv/bin/activate
##     source make_release.sh

# Remove any existing distribution archives
rm -rf dist
mkdir dist

source make_cython.sh

# Generate distribution archives
pip install --upgrade setuptools wheel
python setup.py sdist # bdist_wheel

# Upload
pip install --upgrade twine
python -m twine upload dist/*
