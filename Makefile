test:
	PYTHONPATH=src python -m pytest tests --durations=10 --maxfail=1 --ff

fast-test:
	PYTHONPATH=src python -m pytest tests --durations=10 --hypothesis-profile=coverage

coverage:
	PYTHONPATH=src python -m coverage run --branch --include='src/*.py' -m pytest tests/ --hypothesis-profile=coverage
	python -m coverage report --fail-under=100 --show-missing
