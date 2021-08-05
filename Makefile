requirements_osx:
	brew update
	brew install cmake openmpi

requirements_ubuntu:
	sudo apt-get update
	sudo apt-get install cmake libopenmpi-dev python3-dev zlib1g-dev

env:
	conda env create

.PHONY: requirements_osx requirements_ubuntu install