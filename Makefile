# === USER PARAMETERS

export SRC_DIR=image_drift_generator


export PYTHONPATH='$(shell pwd)'

# == SETUP REPOSITORY AND DEPENDENCIES

dev-sync:
	uv sync --all-extras --cache-dir .uv_cache

prod-sync:
	uv sync --all-extras --no-dev --cache-dir .uv_cache

install-hooks:
	uv run pre-commit install


# === CODE VALIDATION

format:
	uv run ruff format

lint:
	uv run ruff check --fix
	uv run mypy --ignore-missing-imports --install-types --non-interactive --package $(SRC_DIR)

test:
	uv run pytest --verbose --color=yes tests

validate: format lint test