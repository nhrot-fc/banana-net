.PHONY: help clean clean-pyc clean-dist run-examples lint test install dev-install

# Default target executed when no arguments are given to make.
help:
	@echo "Available commands:"
	@echo "  make clean        - Remove all build, test, coverage and Python artifacts"
	@echo "  make clean-pyc    - Remove Python file artifacts (cache, bytecode)"
	@echo "  make clean-dist   - Remove distribution artifacts"
	@echo "  make run-examples - Run all example scripts"
	@echo "  make lint         - Check code style with flake8"
	@echo "  make test         - Run tests"
	@echo "  make install      - Install the package"
	@echo "  make dev-install  - Install the package in development mode"

# Clean everything
clean: clean-pyc clean-dist
	@echo "Cleaned all build artifacts and caches"
	@rm -rf .coverage coverage.xml htmlcov/ .pytest_cache/ .tox/

# Clean Python cache files
clean-pyc:
	@echo "Removing Python cache files..."
	@find . -type d -name "__pycache__" -exec rm -rf {} +
	@find . -type f -name "*.py[cod]" -delete
	@find . -type f -name "*.so" -delete
	@find . -type f -name ".coverage" -delete
	@find . -type d -name ".pytest_cache" -exec rm -rf {} +
	@find . -type d -name ".coverage" -exec rm -rf {} +

# Clean distribution files
clean-dist:
	@echo "Removing distribution files..."
	@rm -rf dist/ build/ .eggs/
	@find . -name "*.egg-info" -exec rm -rf {} +
	@find . -name "*.egg" -exec rm -rf {} +

# Run all examples
run-examples:
	@echo "Running all examples..."
	for example in examples/*.py; do \
		echo "Running $$example..."; \
		python "$$example"; \
	done
	@echo "All examples executed"

# Run linting checks
lint:
	@echo "Running code style checks..."
	@if command -v flake8 >/dev/null; then \
		flake8 banana_net tests examples; \
	else \
		echo "flake8 not installed. Run: pip install flake8"; \
	fi

# Run tests
test:
	@echo "Running tests..."
	@if command -v pytest >/dev/null; then \
		pytest; \
	else \
		echo "pytest not installed. Run: pip install pytest"; \
	fi

# Install the package
install:
	@echo "Installing banana_net..."
	@pip install .

# Install in development mode
dev-install:
	@echo "Installing banana_net in development mode..."
	@pip install -e .