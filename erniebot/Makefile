.DEFAULT_GOAL = dev
files_to_format_and_lint = src examples tests

.PHONY: dev
dev: format lint type-check

.PHONY: format
format:
	python -m black $(files_to_format_and_lint)
	python -m isort $(files_to_format_and_lint)

.PHONY: format-check
format-check:
	python -m black --check --diff $(files_to_format_and_lint)
	python -m isort --check-only --diff $(files_to_format_and_lint)

.PHONY: lint
lint:
	python -m flake8 $(files_to_format_and_lint)

.PHONY: type-check
type-check:
	python -m mypy src
