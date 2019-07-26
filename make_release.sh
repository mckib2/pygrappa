
# Remove any existing distribution archives
rm -rf dist
mkdir dist

# Make sure all Cython business is current
./make_cython.sh

# Generate distribution archives
python3 -m pip install --user --upgrade setuptools wheel
python3 setup.py sdist # bdist_wheel

# Upload
python3 -m pip install --user --upgrade twine
python3 -m twine upload dist/*
