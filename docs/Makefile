# Run from an environment containing the package dependencies and the docs group.
PYTHON ?= python
SPHINXOPTS ?= -W --keep-going

.PHONY: html clean serve

html:
	$(PYTHON) -m sphinx -b html $(SPHINXOPTS) source _build/html

clean:
	rm -rf _build source/_generated

serve: html
	$(PYTHON) -m http.server --bind 127.0.0.1 --directory _build/html 8765
