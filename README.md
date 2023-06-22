# Annotation-Tool

This annotator tool is aimed at annotating hyperspectral images.
It reads H5 files and expects the data to be in a group called 'hypercube'.

![image](https://github.com/Ritchie3/Annotation-Tool/assets/38915268/db267672-3262-40ee-ad73-13fac61f18f4)

The data is expected in the format XYZ, with Z the spectral bands.

![image](https://github.com/Ritchie3/Annotation-Tool/assets/38915268/d0ba13eb-bf08-4953-8275-019495f767f4)

Labelnames should be defined in the label.xlsx file.

After saving, a seperate mask image is created in the H5 file as a 2D array: XY. 
The pixel value is dependant on the layer number: 1,2,3,...

![image](https://github.com/Ritchie3/Annotation-Tool/assets/38915268/41a3545c-1905-4862-ac0e-b19d9035f533)

after the labeling the internal H5 structure will look like this:

![image](https://github.com/Ritchie3/Annotation-Tool/assets/38915268/53aef1f4-be37-4999-9d83-f59aff6806f8)

The H5 file will be renamed to <name>_labeled_hypercube.h5

This annotator version is originally created by Edgar Cardenas. Updated by Ritchie Heirmans. Further developed by bachelor students, with some additional improvements
added functionality:
 - multi label
 - improved GUI
 - Folder selection through GUI
 - Label info from excel sheet:  column header name should be 'label'
 - label_legend created in export h5 file
 

  screenshots taken with HDF Viewer
