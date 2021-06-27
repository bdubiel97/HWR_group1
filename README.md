# HWR_group1
Semestral project for Handwriting Recognition subject at the University of Groningen.
Worked on by Blazej Dubiel (s4525256), Rick de Jonge (s2775832), Ella Collins (s3327469), and Manon Heinhuis (s3378438).

The dependencies used in this pipeline can be obtained by running from our HWR_group1 directory:
```
pip3 install -r /path/to/requirements.txt
```
Additionally, you will need to install the Image processing toolbox version 11.1 add-on in MATLAB. 

Each of the pipeline's subparts have their own READMEs. The pipeline starts with the segmentation, for which the input should be placed within the Segmentation folder, named `input`, preferably as .jpg files. From then on, the directory names are correctly implemented for all code files that require them.

Then, in CharacterClassification, first the file `order_segmented_characters.py` from Helpers should be run to ensure the correct ordering for the writing to file of the classification. Then, `testCharacterClassificationCNN.py` can be run to produce printed output per file in a new folder called `character_results`,  and can be checked here. 

At the same time, for style classification, `test_SOTM.py` can be run to print the output classifications per file in the folder `style_results`. There are several other methods included in the main style classification folder, however, the SOTM outperformed these methods. 

## Single line running
To run the whole pipeline using a single command, open a terminal in the main (`HWR_group1`) folder and call the bash 
file using the following command: `MAIN.sh {path_to_input_files}`