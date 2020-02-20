init:
	pip install -r requirements.txt
setup-test:
	pip install pytest
	pip install pytest-cov
test:
	python setup.py develop
	pytest --cov=fosc --cov-report html

.PHONY: init setup-test test
