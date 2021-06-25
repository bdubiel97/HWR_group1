import splitfolders

#Split augmeted data folder into test, validation, and test data
splitfolders.ratio("monkbrill1ForSplittin", output="monkbrill1SPLIT", seed=1234, ratio=(.6, .2, .2), group_prefix=None) # default values
