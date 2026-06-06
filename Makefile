# Convenience targets for the observability + exploration layer.
# See OBSERVABILITY.md for details.

PY ?= python
DATA ?= data
ARTIFACTS ?= artifacts
REPORT ?= report

.PHONY: setup data preprocess obs explore test all clean

setup:            ## install the observability/exploration dependencies
	$(PY) -m pip install -r requirements-obs.txt

data:             ## unzip the raw CSVs into $(DATA)/
	$(PY) -c "import zipfile; zipfile.ZipFile('dataset/dataset-compressed.zip').extractall('$(DATA)')"

preprocess: data  ## run the leakage-safe pipeline -> $(ARTIFACTS)/
	$(PY) preprocess.py --data $(DATA) --out $(ARTIFACTS)

obs: preprocess   ## run the observability suite -> $(REPORT)/
	$(PY) run_observability.py --data $(DATA) --artifacts $(ARTIFACTS) --out $(REPORT)

explore:          ## launch the interactive dataset explorer
	streamlit run explore/app.py

test:             ## run integrity + golden-value regression tests
	$(PY) -m pytest tests/ -q

all: obs test     ## pipeline + observability + tests

clean:            ## remove generated frames, manifest, and reports
	rm -rf $(ARTIFACTS) $(REPORT) $(DATA)/*.csv
