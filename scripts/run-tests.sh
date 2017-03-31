#!/usr/bin/env bash

set -e -u -x

virtualenv testenv

export PYTHONPATH=src

testenv/bin/pip install -r requirements.txt
testenv/bin/python -m pytest tests --durations=10


python -m coverage run --branch --include='src/*.py' -m pytest tests/ --hypothesis-profile=coverage
python -m coverage report --fail-under=100 --show-missing
