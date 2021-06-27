# Install all required packages exceopt for matlab add-on mentioined in README.
pip3 install -r requirements.txt

# Segmentation using Matlab
cd Segmentation
m_output="output"
m_output_size="[128 128]"
matlab -nodesktop -nosplash -r "main('$1', '$m_output', $m_output_size, false, false);quit"
cd ..

# Character Recognition using python
cd characterRecognition/Helpers
python order_segmented_characters.py
cd ..
python testCharacterClassificationCNN.py
cd ..

# Style classification using python
cd styleClassification/SOTM
python test_SOTM.py
cd ../..
