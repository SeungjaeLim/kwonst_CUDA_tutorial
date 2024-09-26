default: build

help:
	@echo 'Management commands for cu_tutorial:'
	@echo
	@echo 'Usage:'
	@echo '    make build            Build the cu_tutorial project.'
	@echo '    make preprocess       Preprocess step.'
	@echo '    make run              Boot up Docker container.'
	@echo '    make up               Build and run the project.'
	@echo '    make rm               Remove Docker container.'
	@echo '    make stop             Stop Docker container.'
	@echo '    make reset            Stop and remove Docker container.'
	@echo '    make docker-setup     Setup Docker permissions for the user.'

preprocess:
	@echo "Running preprocess step"
	@docker 

build:
	@echo "Building Docker image"
	@docker build . -t cu_tutorial

run:
	@echo "Booting up Docker Container"
	@docker run -it --gpus '"device=1"' --ipc=host --name cu_tutorial -v `pwd`:/workspace cu_tutorial:latest /bin/bash

up: build run

rm: 
	@echo "Removing Docker container"
	@docker rm cu_tutorial

stop:
	@echo "Stopping Docker container"
	@docker stop cu_tutorial

reset: stop rm

docker-setup:
	@echo "Setting up Docker permissions for the current user"
	@sudo groupadd docker || true
	@sudo usermod -aG docker $(USER)
	@newgrp docker
