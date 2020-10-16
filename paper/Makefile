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

separate-paper:
	pdfseparate -f 1 -l 11 main.pdf paper-%02d.pdf
	pdfunite paper-*.pdf paper.pdf
	rm paper-*.pdf

separate-supplementary:
	pdfseparate -f 12 main.pdf supplementary-%02d.pdf
	pdfunite supplementary-*.pdf supplementary.pdf
	rm supplementary-*.pdf

zip:
	sed -i 's/\.\.\///g' main.tex
	cp -r ../plots .
	cp -r ../tikz .
	zip -r latexsource.zip \
			main.tex \
			refs.bib \
			main.bbl \
			neurips_2020.sty \
			tikz \
			plots
	rm -r plots tikz
	echo "Attention!"
	echo "To revert sed changes: git checkout main.tex"
