# WSI_analysis

Multiprocessing Analysis
Read the svs file and create the folders, based on the number of cores, in which there are the tiles in png format.

Analysis

Classification
Open the folders the the previous program have created and classify them, the result are stored in a dictionary in which
are saved also other parameters like the uncertainty, the position of the tile, the shape.
Thanks to the dictionary the method overlay can perform the mask for the prediction and for the 2 kind of uncertainty.
