# Style classification 

## Self-organizing Time Map
The SOTM is trained using the morphed dataset using the helper file `styleImageGenerator.py`. This file takes a class from the training data and uses the morphing tool to morph images from the training data.

### Training
The file `train_SOTM.py` can train the SOTM using the morphed dataset, which must be in a folder called `characters_for_style_classification_morph`. The file saves three separate files: `SOTM_arc.p`, `SOTM_has.p` and `SOTM_her.p`. 

### Testing
The file `test_SOTM.py` can be used to test the SOTM on documents from the different periods. To test the SOTM, the zip files, `SOTM_arc.p.zip`, `SOTM_has.p.zip` and `SOTM_her.p.zip`, must be decompressed first. Running the file will create a directory called `style_results` which will include a result for each document that is being tested. These files will be named `document_style.txt`, where "document" is substituted with the name of each file that is classified.

## Self-organizing Map


## CNN
### Training
The file `train_model.py` can train a model on a dataset supplied in the `training_input` folder. 
The images in this folder should be pre-processed using the `pre_processing.py` file first.
The resulting model is saved in the folder `trained_models`.

### Validation
Using this model, you can run the `validate.py` file to determine the time period of a set of input images. 
The default folder that is taken as the input is the `Segmentation/output` folder. 
This folder should hold a folder with images for each input image.

The results of this validation is printed to the screen, and saved in a file `results.txt`.