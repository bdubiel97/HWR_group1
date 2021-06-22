# Segmentation
## Character segmentation example

Run the main.m file using MATLAB. It will search for input images in a directory named `input`, taking all images found.
It has 3 parameters:

-|Parameter|Description
--- | --- | ---
1. | debug            | `1` in case you want to pause between every input file, else `0`
2. | output size      | Probably `[64 64]`
3. | showImages       | `1` if you want MATLAB to plot figures, else `0`
Example: `main(0, [64 64], 0)`  

The output will be stored into a folder "output", where a folder "failures" holds images that exceed the specified output size.