import pandas as pd
import numpy as np


final_pred = []

ids_       = pd.read_csv('./submission_pac2018_run1.csv') ['PAC_ID'].as_matrix()

for id_ in ids_:
	data_all   = []
	for i in range(10):
		df = pd.read_csv('./submission_pac2018_run' + str(i+1) + '.csv')
		data_all.append(df[df['PAC_ID'] == id_]['Prediction'].values)

	unique,counts = np.unique(data_all, return_counts=True)
	final_pred.append(unique[np.argmax(counts)]) ### one max done
	print (id_, unique[np.argmax(counts)])

sub               = pd.DataFrame()
sub['PAC_ID']     = ids_
sub['Prediction'] = final_pred
sub.to_csv('final_submission_pac2018.csv')