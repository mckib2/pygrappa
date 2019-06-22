
# Remove any existing distribution archives
rm -rf dist

# Generate distribution archives
python3 -m pip install --user --upgrade setuptools wheel
python3 setup.py sdist bdist_wheel

# Upload
python3 -m pip install --user --upgrade twine
python3 -m twine upload dist/*
