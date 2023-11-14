PYTHON = python3
PIP = pip

ENV = secrec
REFERENCE_DIR = reference

.PHONY: setup

setup:
	$(PYTHON) -m virtualenv $(ENV)
	. $(ENV)/bin/activate;	\
	pip install -r requirements.txt;

activate:
	. $(ENV)/bin/activate;

# Run all unittests
# run "python3 -m unittest --help" for more options
# on how to run unit tests
test:
	$(PYTHON) -m unittest -v

clean-env:
	rm -rf $(ENV)
	rm -rf __pycache__
	rm -rf .pytest_cache
	@echo "Environment cleared.  Run 'make setup' again to create it."

