# HWR_group1
Semestral project for Handwriting Recognition subject at the University of Groningen.
Worked on by Blazej Dubiel (s4525256), Rick de Jonge (s2775832), Ella Collins (s3327469), and Manon Heinhuis (s3378438).

Each of the pipeline's subparts have their own READMEs. The pipeline starts with the segmentation, for which the input should be placed within the Segmentation folder, named "input", preferably as .jpg files. From then on, the directory names are correctly implemented for all code files that require them.

Then, in CharacterClassification, first the file "order_segmented_characters_15.48.18.py" from Helpers should be run to ensure the correct ordering for the writing to file of the classification. Then, "testCharacterClassificationCNN.py" can be run to produce printed output per file in a new folder called "character_results",  and can be checked here. 

At the same time, for style classification, test_SOTM.py can be run to print the output classifications per file in the folder "style_results".