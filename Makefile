setup-test:
	pip install -r requirements-dev.txt
test:
	python setup.py develop
	pytest --cov=fosc --cov-report html

.PHONY: init setup-test test
