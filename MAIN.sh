# Segmentation using Matlab
cd Segmentation
m_output="output"
m_output_size="[128 128]"
matlab -nodesktop -nosplash -r "main('$1', '$m_output', $m_output_size, false, false);quit"
cd ..

# Character Recognition using python

# Style classification using python