TARGETS = main.pdf
SOURCES = $(shell find . -name '*.tex')

main.pdf: main.tex $(SOURCES)
	pdflatex main.tex
	bibtex main
	pdflatex main.tex
	pdflatex main.tex

clean:
	rm main.pdf
	rm main.aux
	rm main.bbl
	rm main.blg
	rm main.log
	rm main.out

open:
	evince $(TARGETS) &
