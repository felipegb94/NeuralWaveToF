pdf:
	pdflatex NeuralArchitecture_SinglePixelDepth
	convert -density 150 NeuralArchitecture_SinglePixelDepth.pdf -quality 90 NeuralArchitecture_SinglePixelDepth.png

all:
	make pdf; make pdf; make pdf
clean:
	rm *.aux *.bbl *.blg *.log;
