# Segmentation
## Character segmentation example

Run the main.m file using MATLAB. It will search for input images in a directory named `input`, taking all images found.
It has 3 parameters:

-|Parameter|Description|
--- | --- | ---|
1. | debug            | `true` in case you want to pause between every input file, else `false`|
2. | output size      | According to the given test images, output size should be set to `[128 128]`|
3. | showImages       | `true` if you want MATLAB to plot figures, else `false`|
Example: `main(0, [128 128], 0)`

The output will be stored into a folder `output`, where a folder `output_failures` holds the images that were not 
classified as noise initially, but have been discarded.