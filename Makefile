.ONESHELL:
# Variables
PATH_TO_CONFIG := "churner/ml/config-ml.yaml"
# Train a model
.PHONY: train
train:
	python churner/ml/train.py ${PATH_TO_CONFIG}
# Build docker image
build:
	docker build -t churn_api churner/app/
# Run churn api container
run:
	docker run -p 5555:5555 churn_api