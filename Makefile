# Makefile for A star trellis
SHELL := /bin/bash

# You can set these variables from the commandline.
VERSION=$(shell python setup.py --version)

distclean:
	pip uninstall A_Star_Trellisr
	rm -r dist/ build/

./dist/A_Star_Trellis-${VERSION}-py3-none-any.whl:
	python ./setup.py sdist bdist_wheel

#install:
#	pip install -e .
install: ./dist/A_Star_Trellis-${VERSION}-py3-none-any.whl # pip install
	pip install --upgrade ./dist/A_Star_Trellis-${VERSION}-py3-none-any.whl

%: Makefile
