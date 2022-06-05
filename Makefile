setup-test:
	virtualenv -p `which python3` venv	
	. venv/bin/activate; \
	pip install -r requirements-old.txt; \
	pip install -r requirements-dev.txt; \
	deactivate
	virtualenv -p `which python3` venv2	
	. venv2/bin/activate; \
	pip install -r requirements.txt; \
	pip install -r requirements-dev.txt; \
	deactivate
test:
	. venv/bin/activate; \
		pytest --cov=fosc --cov-report html tests/test_persist.py
	. venv2/bin/activate; \
		pytest --cov=fosc --cov-report html tests/test_persist.py
	coverage report -m
.PHONY: init setup-test test
