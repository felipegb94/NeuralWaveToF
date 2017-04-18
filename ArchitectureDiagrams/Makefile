pdf:
	pdflatex NeuralArchitecture_BlackBox
	convert -density 150 NeuralArchitecture_BlackBox.pdf -quality 90 NeuralArchitecture_BlackBox.png

all:
	make pdf; make pdf; make pdf
clean:
	rm *.aux *.bbl *.blg *.log;
