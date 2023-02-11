SHELL=/bin/bash
PATH := .venv/bin:$(PATH)
export TEST?=./tests
export ENV?=dev


# Makefile

# to create virtual environment and install dependencies
install:
	@( \
		if [ ! -d .venv ]; then python3 -m venv --copies .venv; fi; \
		source .venv/bin/activate; \
		pip install -qU pip; \
		pip install -r requirements-dev.txt; \
		pip install -r requirements.txt; \
	)

# to train and registry a new model
model-train:
	@source .venv/bin/activate; \
	cd src/mlflow; \
	python train_model.py

model-serve:
	@source .venv/bin/activate; \
	cd src/mlflow; \
	python model_serve.py


# to make predictions
model-predict:
	@source .venv/bin/activate; \
	cd src/mlflow; \
	python predict.py 	


# to run the MLFlow UI
mlflow-ui:
	@source .venv/bin/activate; \
	cd src/mlflow; \
	mlflow server --backend-store-uri sqlite:///mlruns.db -h 0.0.0.0 -p 5000 

autoflake:
	@autoflake . --check --recursive --remove-all-unused-imports --remove-unused-variables --exclude .venv;

black:
	@black . --check --exclude '.venv|build|target|dist|.cache|node_modules';

isort:
	@isort . --check-only;

lint: black isort autoflake

# to clean code and remove all temporary files
lint-fix:
	@black . --exclude '.venv|build|target|dist';
	@isort .;
	@autoflake . --in-place --recursive --exclude .venv --remove-all-unused-imports --remove-unused-variables;