pdf:
	pdflatex NeuralArchitecture_SinglePixelDepth2
	convert -density 150 NeuralArchitecture_SinglePixelDepth2.pdf -quality 90 NeuralArchitecture_SinglePixelDepth2.png

all:
	make pdf; make pdf; make pdf
clean:
	rm *.aux *.bbl *.blg *.log;
