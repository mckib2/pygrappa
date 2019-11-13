#!/usr/bin/env bash

for f in pygrappa/examples/*.py; do
    echo "$f"
    python -m pygrappa.examples.$(basename "$f" .py) || break
done
