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

zip:
	cp main.tex main.tex.backup
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
	mv main.tex.backup main.tex
	echo "Attention!"
	echo "To revert sed changes: git checkout main.tex"

