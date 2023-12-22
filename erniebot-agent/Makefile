.DEFAULT_GOAL = dev
files_to_format_and_lint = src examples tests

.PHONY: dev
dev: format lint type_check

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

.PHONY: test
test:
	python -m pytest tests/unit_tests

.PHONY: test_coverage
test_coverage:
	python -m pytest tests/unit_tests --cov erniebot_agent --cov-report xml:coverage.xml
