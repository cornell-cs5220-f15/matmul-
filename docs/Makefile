LATEX=pdflatex
BIBTEX=bibtex

PDFFILES=group018.pdf

all: clean $(PDFFILES)

$(PDFFILES): %.pdf: %.tex group018.tex
	$(LATEX) $<
	$(LATEX) $<
	$(LATEX) $<
	$(LATEX) $<

clean:
	rm -f *.aux *.log *.dvi $(PAPER).pdf *.lot *.lof *.toc *.blg *.bbl *.ent *.out
