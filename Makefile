# Reproduce everything: data, models, figures, slides and the dashboard.
#
#   make setup    create the virtualenv and install the package
#   make all      run the full pipeline end to end
#   make slides   rebuild the presentation
#   make serve    preview the dashboard locally
#   make test     run the test suite
#   make lint     check style
#   make clean    remove generated artefacts (keeps downloaded data)

PY      := .venv/bin/python
PIP     := .venv/bin/pip
SRC     := src
PKG     := src/fhew
FIGURES := outputs/figures
RAW     := data/raw/iod2019/iod2019_lsoa_all.csv
PANEL   := data/processed/panel.parquet
DESIGN  := data/processed/design.parquet
LSOA    := data/processed/colchester_lsoa.parquet

.PHONY: all setup data panel model figures facts dashboard profile slides serve test lint clean distclean

all: slides dashboard profile

# The package is installed editable, so `import fhew` works from the stage
# scripts, the tests and a notebook without anything touching sys.path.
setup: .venv/bin/activate
.venv/bin/activate: pyproject.toml requirements-dev.txt
	python3 -m venv .venv
	$(PIP) install --quiet --upgrade pip
# The network only ever runs on CPU, and torch's default wheels carry a CUDA
# runtime that is orders of magnitude larger than anything here needs. Taking
# the CPU wheel first means the editable install below finds the requirement
# already satisfied. The leading '-' lets this fall through to the default
# index on platforms that publish no CPU wheel.
	-$(PIP) install --quiet torch --index-url https://download.pytorch.org/whl/cpu
	$(PIP) install --quiet -r requirements-dev.txt
	@touch $@

# --- data ------------------------------------------------------------------
data: $(RAW)
$(RAW): $(SRC)/00_download.py $(PKG)/sources.py | setup
	$(PY) $(SRC)/00_download.py

# Parsing the seven .ods releases takes a couple of minutes, so each year is
# cached and the stage is a no-op on reruns.
data/interim/hclic_2024-25.parquet: $(SRC)/01_extract_hclic.py $(PKG)/hclic.py $(RAW)
	$(PY) $(SRC)/01_extract_hclic.py

panel: $(PANEL)
$(PANEL): $(SRC)/02_build_panel.py $(PKG)/panel.py $(PKG)/config.py \
          data/interim/hclic_2024-25.parquet
	$(PY) $(SRC)/02_build_panel.py

$(LSOA): $(SRC)/03_colchester_lsoa.py $(PKG)/neighbourhoods.py $(RAW)
	$(PY) $(SRC)/03_colchester_lsoa.py

# --- analysis --------------------------------------------------------------
model: $(DESIGN)
$(DESIGN): $(SRC)/04_model.py $(PKG)/features.py $(PKG)/models.py \
           $(PKG)/evaluation.py $(PKG)/mlp.py $(PANEL)
	$(PY) $(SRC)/04_model.py

figures: $(FIGURES)/fig_trend.pdf
$(FIGURES)/fig_trend.pdf: $(SRC)/05_figures.py $(PKG)/plotting.py $(DESIGN) $(LSOA)
	$(PY) $(SRC)/05_figures.py

facts: slides/facts.tex
slides/facts.tex: $(SRC)/06_facts.py $(PKG)/export.py $(DESIGN) $(LSOA)
	$(PY) $(SRC)/06_facts.py

dashboard: docs/data/ranking.json
docs/data/ranking.json: $(SRC)/07_dashboard_data.py $(PKG)/export.py $(DESIGN) $(LSOA)
	$(PY) $(SRC)/07_dashboard_data.py

profile: DATASET.md
DATASET.md: $(SRC)/08_data_profile.py $(PKG)/profiling.py $(DESIGN) $(LSOA)
	$(PY) $(SRC)/08_data_profile.py

# --- outputs ---------------------------------------------------------------
slides: slides/presentation.pdf
slides/presentation.pdf: slides/presentation.tex slides/facts.tex $(FIGURES)/fig_trend.pdf
	cd slides && latexmk -pdf -interaction=nonstopmode -halt-on-error presentation.tex

serve: docs/data/ranking.json
	@echo "Dashboard at http://localhost:8000  (Ctrl-C to stop)"
	@cd docs && python3 -m http.server 8000

# --- quality ---------------------------------------------------------------
test: | setup
	.venv/bin/pytest tests

lint: | setup
	.venv/bin/ruff check $(SRC) tests

# --- housekeeping ----------------------------------------------------------
clean:
	rm -rf outputs/figures/* outputs/tables/* outputs/models/*
	rm -f data/processed/*.parquet slides/facts.tex
	cd slides && latexmk -C >/dev/null 2>&1 || true

distclean: clean
	rm -rf data/raw data/interim .venv src/fhew.egg-info
