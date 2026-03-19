
.PHONY: v3 install venv indicators report tex theory clean clean-tex

VENV ?= .venv
PYTHON ?= python3

LATEX ?= pdflatex
BIBER ?= biber
BIBTEX ?= bibtex

TEXDIR ?= tex
THEORY ?= theory

VENV_PY := $(VENV)/bin/python
VENV_PIP := $(VENV)/bin/pip
STAMP := $(VENV)/.installed

v3: report tex

report: indicators
	$(VENV_PY) scripts/03_make_report.py

tex: $(TEXDIR)/$(THEORY).pdf

theory: tex

$(TEXDIR)/$(THEORY).pdf: $(TEXDIR)/$(THEORY).tex
	cd $(TEXDIR) && $(LATEX) -interaction=nonstopmode -halt-on-error $(THEORY).tex
	@if [ -f "$(TEXDIR)/$(THEORY).bcf" ]; then \
		cd $(TEXDIR) && $(BIBER) $(THEORY); \
	elif [ -f "$(TEXDIR)/$(THEORY).bib" ]; then \
		cd $(TEXDIR) && $(BIBTEX) $(THEORY); \
	fi
	cd $(TEXDIR) && $(LATEX) -interaction=nonstopmode -halt-on-error $(THEORY).tex
	cd $(TEXDIR) && $(LATEX) -interaction=nonstopmode -halt-on-error $(THEORY).tex

indicators: install
	$(VENV_PY) scripts/02_compute_indicators.py

install: $(STAMP)

venv: $(VENV_PY)

$(VENV_PY):
	$(PYTHON) -m venv $(VENV)

$(STAMP): requirements.txt | $(VENV_PY)
	$(VENV_PIP) install -U pip
	@if [ -f constraints.txt ]; then \
		$(VENV_PIP) install -r requirements.txt -c constraints.txt; \
	else \
		$(VENV_PIP) install -r requirements.txt; \
	fi
	@touch $(STAMP)

clean:
	rm -f site/report.html site/archive.json site/feed.xml site/sitemap.xml site/robots.txt
	rm -f site/indicators.csv site/indicators_us.csv site/indicators_eu.csv

clean-tex:
	rm -f $(TEXDIR)/*.aux $(TEXDIR)/*.bbl $(TEXDIR)/*.blg $(TEXDIR)/*.bcf $(TEXDIR)/*.run.xml $(TEXDIR)/*.log $(TEXDIR)/*.out
