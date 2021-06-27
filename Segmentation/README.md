# Segmentation
## Character segmentation example
* In order to run the code, Image Processing Toolbox must be installed within MATLAB environment 

Run the main.m file using MATLAB. It will search for input images in a directory named `input`, taking all images found.
It has 3 parameters:

Nr. |Parameter          |Description|
--- | ---               | ---|
1.  | input_folder      | The path to the input images.|
2.  | output_fodler     | The path to where the results will be stored.|
3.  | output size       | The output size should be set to `[128 128]` to conform with the input to the later stages of the pipeline.|
4.  | debug             | `true` in case you want to pause between every input file, else `false`.|
5.  | showImages        | `true` if you want MATLAB to plot figures, else `false`.|

To segment the characters ready for the subsequent parts in the pipeline, please use `main(0, [128 128], 0)` to run the 
program using Matlab.

The output will be stored into a folder `output`, where the characters are organized into folders based on the input 
image they were segmented from. The name of the image holds some extra information about the location and size of the 
image, which is needed to sort them later. An extra folder `output_failures` holds the images that were not classified 
as noise initially, but have been discarded.
