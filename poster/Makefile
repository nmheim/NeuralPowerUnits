TARGETS = main.pdf
SOURCES = $(shell find . -name '*.tex')

main.pdf: main.tex $(SOURCES)
	xelatex main.tex

clean:
	rm main.pdf
	rm main.aux
	rm main.log

open:
	evince $(TARGETS) &

png:
	convert -density 200 -quality 90 main.pdf main.png
