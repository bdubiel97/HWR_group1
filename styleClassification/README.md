# Style classification 
We have included three different approaches to the style classification problem in this section. Of these approaches, 
the best results were obtained using the SOTM. The other two approaches, SOM and CNN, have been included in the 
`OtherAttempts` folder.

## Self-organizing Time Map
The SOTM is trained using the morphed dataset using the helper file `styleImageGenerator.py`. This file takes a class from the training data and uses the morphing tool to morph images from the training data.

### Training
The file `train_SOTM.py` can train the SOTM using the morphed dataset, which must be in a folder called `characters_for_style_classification_morph`. The file saves three separate files: `SOTM_arc.p`, `SOTM_has.p` and `SOTM_her.p`. 

### Testing
The file `test_SOTM.py` can be used to test the SOTM on documents from the different periods. Running the file will create a directory called `results` in the main directory which will include a result for each document that is being tested. These files will be named `document_style.txt`, where "document" is substituted with the name of each file that is classified.

## Self-organizing Map
The SOM is trained using the morphed dataset using the helper file `styleImageGenerator.py`. This file takes a class from the training data and uses the morphing tool to morph images from the training data.

### Training 
The file `train_SOM.py` can train the SOM using the morphed dataset, which must be in a folder called `characters_for_style_classification_morph`. The file saves the SOM as `som.py` and provides a classification accuracy for the characters in the testing data.

### Testing 
The file `test_SOM.py` can be used to test the SOM on the documents from the different periods. The training data, which must be in a folder called `characters_for_style_classification_morph`, is also required to run this file. Running the file will create a directory called `results` in the main directory which will include a result for each document that is being tested. These files will be named `document_style.txt`, where "document" is substituted with the name of each file that is classified.

## CNN
### Training
The file `train_cnn.py` can train a model on a dataset supplied in the `training_input` folder. 
The images in this folder should be pre-processed using the `pre_processing.py` file first.
The resulting model is saved in the folder `trained_models`.

### Testing
Using this model, you can run the `test_cnn.py` file in the `CNN` directory to determine the time period of a set of input
images. The default folder that is taken as the input is the `Segmentation/output` folder. This folder should hold a 
folder with images for each input image. The results of this validation is printed to the screen, and saved in a file 
`results.txt`.
