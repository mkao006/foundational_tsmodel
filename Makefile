env:
	pyenv virtualenv 3.11 foundational_tsmodel
	pyenv local foundational_tsmodel

install:
	poetry install

run:
	python main.py


## Remove all build, test, coverage and Python artifacts
clean:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +
