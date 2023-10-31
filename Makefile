# Makefile for ErnieBot Agent
#
# 	GitHb: https://github.com/PaddlePaddle/Ernie-Bot-Agent
# 	Author: Paddle Team https://github.com/PaddlePaddle
#

.PHONY: all
all : lint test
# # # # # # # # # # # # # # # Format Block # # # # # # # # # # # # # # # 

format:
	pre-commit run isort
	pre-commit run black

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # Lint Block # # # # # # # # # # # # # # # 

.PHONY: lint
lint:
	$(eval modified_py_files := $(shell python scripts/get_modified_files.py $(check_dirs)))
	@if test -n "$(modified_py_files)"; then \
		echo ${modified_py_files}; \
		pre-commit run --files ${modified_py_files}; \
	else \
		echo "No library .py files were modified"; \
	fi	

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # # # # # Test Block # # # # # # # # # # # # # # # 

.PHONY: test
test: unit-test

unit-test:
	PYTHONPATH=$(shell pwd) pytest -v \
		-n auto \
		--durations 20 \
		--cov erniebot_agent \
		--cov-report xml:coverage.xml

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

.PHONY: install
install:
	pip install -r requirements-dev.txt
	pip install -r requirements.txt
	pre-commit install


.PHONY: deploy
deploy:
	# install related package
	make install
	# build
	python3 setup.py sdist bdist_wheel
	# upload
	twine upload --skip-existing dist/*
