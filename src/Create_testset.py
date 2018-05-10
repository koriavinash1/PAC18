import pandas as pd
import numpy as np

## load all csv files
paths = []
labels = []
scanners = []
gender = []

s1f = "../processed_data/scanner_1_female_train_test_split.csv"
slf = pd.read_csv(slf)[pd.read_csv(slf)['Testing']]
paths.extend(slf['Volume_Path'].as_matrix())
labels.extend(slf['Labels'].as_matrix())
scanners.extend([1]*len(s1f))
gender.extend(['female']*len(s1f))

s2f = "../processed_data/scanner_2_female_train_test_split.csv"
s2f = pd.read_csv(s2f)[pd.read_csv(s2f)['Testing']]
paths.extend(s2f['Volume_Path'].as_matrix())
labels.extend(s2f['Labels'].as_matrix())
scanners.extend([1]*len(s2f))
gender.extend(['female']*len(s2f))

s3f = "../processed_data/scanner_3_female_train_test_split.csv"
s3f = pd.read_csv(s3f)[pd.read_csv(s3f)['Testing']]
paths.extend(s3f['Volume_Path'].as_matrix())
labels.extend(s3f['Labels'].as_matrix())
scanners.extend([1]*len(s3f))
gender.extend(['female']*len(s3f))

s1m = "../processed_data/scanner_1_male_train_test_split.csv"
slm = pd.read_csv(slm)[pd.read_csv(slm)['Testing']]
paths.extend(slm['Volume_Path'].as_matrix())
labels.extend(slm['Labels'].as_matrix())
scanners.extend([1]*len(s1m))
gender.extend(['female']*len(s1m))

s2m = "../processed_data/scanner_2_male_train_test_split.csv"
s2m = pd.read_csv(s2m)[pd.read_csv(s2m)['Testing']]
paths.extend(s2m['Volume_Path'].as_matrix())
labels.extend(s2m['Labels'].as_matrix())
scanners.extend([1]*len(s2m))
gender.extend(['female']*len(s2m))

s3m = "../processed_data/scanner_3_male_train_test_split.csv"
s3m = pd.read_csv(s3m)[pd.read_csv(s3m)['Testing']]
paths.extend(s3m['Volume_Path'].as_matrix())
labels.extend(s3m['Labels'].as_matrix())
scanners.extend([1]*len(s3m))
gender.extend(['female']*len(s3m))

##############################################
# create csv
