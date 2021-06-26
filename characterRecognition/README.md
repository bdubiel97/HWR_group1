# Character Classification
README by Manon
## Character Classification Example

First, the segmented data needs to be ordered to write to file correctly. None of the following python files require input arguments. To do this, use order_segmented_characters_15.48.18.py from the Helpers folder (run usinig python). This will then create a folder named sorted_output in CharacterClassification. The data is then ready to be used by testCharacterClassificationCNN.py, using python, which will write the classification of characters to a file per original image file in the character_results folder. This are the final results for character classification in the pipeline. 

## Character Classification Background
To train the model from scratch, you will need to include training  data. These can be obtained by including the provided monkbrill2 data in the Helpers folder. Then, using imageGenerator.py, more and more varied data will be generated in the same folder provided as input folder. To create training, validation and testing data, use TrainTestSplitting.py, which will create the folder splitCharacterData with three subfolders: train, val, and test. These can the be used by 