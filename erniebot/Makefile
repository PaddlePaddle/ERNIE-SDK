.DEFAULT_GOAL = format lint type_check
files_to_format_and_lint = src examples tests

.PHONY: format
format:
	python -m black $(files_to_format_and_lint)
	python -m isort --filter-files $(files_to_format_and_lint)

.PHONY: lint
lint:
	python -m flake8 $(files_to_format_and_lint)

.PHONY: type_check
type_check:
	python -m mypy src
