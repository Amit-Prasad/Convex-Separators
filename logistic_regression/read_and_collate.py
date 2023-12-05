import pandas as pd
import ast

datasets = ["churn", "telco_churn", "santander_sub", "covtype_bin_sub", "spambase", "shoppers", "diabetes", "breast_cancer", "ionosphere", "philippine"]
output = []
for data_name in datasets:
	with open("temp_" + "lr_" + data_name + ".txt") as f:
		data = f.read()
	output_dict = ast.literal_eval(data)
	df=pd.Series(data).to_frame()
	df.to_csv('lr_' + str(data_name) + '.csv', index=False)
	output.append(output_dict)

df = pd.DataFrame.from_records(output)
df.to_csv('all' + '.csv', index=False)
