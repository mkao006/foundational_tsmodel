env:
	pyenv virtualenv 3.11 foundational_tsmodel
	pyenv local foundational_tsmodel

install:
	poetry install

run:
	python main.py
