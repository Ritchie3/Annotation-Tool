# Annotation-Tool

This annotator tool is aimed at annotating hyperspectral images.
It reads H5 files and expects the data to be in a group called 'hypercube'.
The data is expected in the format XYZ, with Z the spectral bands.

Labelnames should be defined in the label.xlsx file.
A seperate mask image is created with 1 band: X x Y x 1. 
The pixel value is dependant on the layer number: 1,2,3,...

after the labeling the internal H5 structure will look like this:
![image](https://github.com/Ritchie3/Annotation-Tool/assets/38915268/53aef1f4-be37-4999-9d83-f59aff6806f8)



This annotator version is originally created by Edgar Cardenas. Updated by Ritchie Heirmans. Further developed by bachelor students, with some additional improvements
added functionality:
 - multi label
 - improved GUI
 - Folder selection through GUI
 - Label info from excel sheet:  column header name should be 'label'
 - label_legend created in export h5 file
 

  
