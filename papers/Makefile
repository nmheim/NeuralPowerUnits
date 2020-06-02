TARGETS = main.pdf
SOURCES = $(shell find . -name '*.tex')

main.pdf: main.tex $(SOURCES)
	lualatex main.tex
	bibtex main
	lualatex main.tex
	lualatex main.tex

clean:
	rm main.pdf
	rm main.aux
	rm main.bbl
	rm main.blg
	rm main.log
	rm main.out

open:
	evince $(TARGETS) &
