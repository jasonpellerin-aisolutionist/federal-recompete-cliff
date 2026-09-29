# Locally, keys come from Infisical; CI passes FEDSPEND_API_KEY directly and sets SECRETS="".
SECRETS ?= scripts/with-secrets.sh
WORKERS ?= 4
YEARS ?= 2012 2013 2014 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 2025 2026

.PHONY: setup download backfill refresh fedspend build label audit-sample features train score app lint test all weekly digest

setup:
	uv sync

download:
	uv run python -m recompete.download --years $(YEARS) --workers $(WORKERS)

# One time: every award modified in the last two years, which picks up active contracts signed
# before FY2012 (long-running M&O and support contracts the fiscal-year pulls cannot see).
backfill:
	uv run python -m recompete.download --backfill-days 730 --workers $(WORKERS)

# New awards and every award modified in the last two weeks (extensions, new obligations).
refresh:
	uv run python -m recompete.download --modified-days 14

fedspend:
	$(SECRETS) uv run python -m recompete.fedspend

build:
	uv run python -m recompete.build

label:
	uv run python -m recompete.label

# Draws a fresh 100-pair sample for blind review; overwrites docs/label_audit_sample.csv.
audit-sample:
	uv run python -m recompete.label --audit-sample

features:
	uv run python -m recompete.features

train:
	uv run python -m recompete.model

score:
	uv run python -m recompete.score

app:
	uv run streamlit run app/streamlit_app.py

lint:
	uv run ruff check src app tests
	uv run ruff format --check src app tests

test:
	uv run pytest -q

all: download backfill fedspend build label features train score

# Weekly job (GitHub Actions, Mondays): new awards and Fed-Spend radar, rescore with the
# evaluated model. Retraining is a deliberate, reviewed step, not part of the weekly refresh.
weekly: refresh fedspend build label features score

digest:
	uv run python -m recompete.digest
