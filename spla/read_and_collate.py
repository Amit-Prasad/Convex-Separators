import pandas as pd
import ast

#datasets = ["circle", "ellipse", "square", "triangle", "pentagon", "hexagon", "sphere_7D", "ellipsoid_7D", "polyhedra_7D", "polyhedra_7D_sparse", "polyhedra_1500D", "polyhedra_1500D_sparse"]
datasets = ["ionosphere", "breast_cancer", "diabetes", "churn", "telco_churn", "santander_sub", "covtype_bin_sub", "spambase", "shoppers", "philippine"]

output = []
for data_name in datasets:
	with open("temp_" + "spla_" + data_name + ".txt") as f:
    		data = f.read()
	output_dict = ast.literal_eval(data)
	#df = pd.DataFrame(output_dict)
	df=pd.Series(data).to_frame()
	df.to_csv('spla_' + str(data_name) + '.csv', index=False)
	output.append(output_dict)

df = pd.DataFrame.from_records(output)
df.to_csv('all_spla' + '.csv', index=False)
