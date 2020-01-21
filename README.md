# WSI_analysis

Multiprocessing Analysis
Read the svs file and create the folders, based on the number of cores, in which there are the tiles in png format.

Analysis

Classification
Open the folders the the previous program have created and classify them, the result are stored in a dictionary in which
are saved also other parameters like the uncertainty, the position of the tile, the shape.
Thanks to the dictionary the method overlay can perform the mask for the prediction and for the 2 kind of uncertainty.

Ui Tool
The ui use the PyQt5 library, it's a cross platform toolkit that works on Windows, Linux, Mac, IOs and also Android.
The tool gives different options:
1) Visualize the complete image svs selected
2) Zoom in and Zoom out options
3) Deepzoom visualizer, that gives the ability to the user to zoom in the svs file at maximum resolution, this is possible
thanks to a local web browser
4) Analise the svs file with a Deep Net