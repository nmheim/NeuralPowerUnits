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

separate-paper:
	pdfseparate -f 1 -l 11 main.pdf paper-%02d.pdf
	pdfunite paper-*.pdf paper.pdf
	rm paper-*.pdf

separate-supplementary:
	pdfseparate -f 12 main.pdf supplementary-%02d.pdf
	pdfunite supplementary-*.pdf supplementary.pdf
	rm supplementary-*.pdf
