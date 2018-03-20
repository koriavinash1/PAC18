# PAC18
sMRI based depression classification

### Directory Structure 

> "./models" : for all saved models (in gitignore)

> "./raw_data/pac2018_data" : for training data (in gitignore)

> "./processed_data/hdf5_files" : for training data after pre processing if required (in gitignore)\



### Packages Required

> keras (updated)

> jupyter

> pytorch

> pandas

> simple ITK

> nibabel


### About ./src/*

+ DataAugment : Contains all required functions for 3D data augmentations

+ DataGenerator : Contains function to load data batch wise..

+ DensenetModel : Contains 3D Densenet model.....

+ Trainer_Tester : Contains train and test class with all train and test strategy


### Working....

+ First use Create_Data.ipynb to create prpcessed data

+ Use ./src/main.py to train/test model

<hr>

If any comments or information required, pull requests/issues are Welcomed....

Thankyou