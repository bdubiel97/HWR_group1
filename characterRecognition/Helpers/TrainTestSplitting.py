import splitfolders

input_folder = "monkbrill2" #Change to augmented data folder

#Split augmeted data folder into test, validation, and test data
splitfolders.ratio(input_folder, output="splitCharacterData", seed=1234, ratio=(.6, .2, .2), group_prefix=None) # default values
