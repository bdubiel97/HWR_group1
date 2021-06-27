# Style classification 
## CNN
### Training
The file `train_model.py` can train a model on a dataset supplied in the `training_input` folder. 
The images in this folder should be pre-processed using the `pre_processing.py` file first.
The resulting model is saved in the folder `trained_models`.

### Validation
Using the model, of which the latest model saved in `trained_models/latest` is loaded by default, you can run the 
`validate.py` file to determine the time period of a set of input images. The default folder that is taken as the input 
is the `Segmentation/output` folder. This folder should hold a folder with images for each input image.

The results of this validation is printed to the screen, and saved in a file `results.txt`.