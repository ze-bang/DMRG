# Build the pedagogical companion PDF.
#
# Requires a TeX Live installation with pdflatex and (optionally) latexmk.
# Usage:
#   make            # build the PDF
#   make watch      # rebuild on every save (needs latexmk)
#   make clean      # remove auxiliary files
#   make distclean  # also remove the PDF

DOC      := dmrg_pedagogical
SRC      := $(DOC).tex
PDF      := $(DOC).pdf
LATEXMK  := latexmk -pdf -halt-on-error -interaction=nonstopmode
PDFLATEX := pdflatex -halt-on-error -interaction=nonstopmode

.PHONY: all watch clean distclean

all: $(PDF)

$(PDF): $(SRC)
	@if command -v latexmk >/dev/null 2>&1; then \
		$(LATEXMK) $(SRC); \
	else \
		$(PDFLATEX) $(SRC); \
		$(PDFLATEX) $(SRC); \
	fi

watch: $(SRC)
	latexmk -pdf -pvc -halt-on-error -interaction=nonstopmode $(SRC)

clean:
	@rm -f $(DOC).aux $(DOC).log $(DOC).out $(DOC).toc \
	       $(DOC).fls $(DOC).fdb_latexmk $(DOC).synctex.gz \
	       $(DOC).bbl $(DOC).blg $(DOC).lof $(DOC).lot $(DOC).nav \
	       $(DOC).snm $(DOC).vrb

distclean: clean
	@rm -f $(PDF)
