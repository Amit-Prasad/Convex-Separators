import pandas as pd
import subprocess
import ast
#datasets = ["circle", "ellipse", "square", "triangle", "pentagon", "hexagon", "sphere_7D", "ellipsoid_7D", "polyhedra_7D", "polyhedra_7D_sparse", "polyhedra_1500D", "polyhedra_1500D_sparse"]
datasets = ["ionosphere", "breast_cancer", "diabetes", "churn", "telco_churn", "santander_sub", "covtype_bin_sub", "spambase", "shoppers", "philippine"]

for data_name in datasets:
    f = open("temp_" + "spla_" + data_name + ".txt", "w")
    subprocess.run(["python3 spla.py " + data_name], stderr=f, stdout=f, shell=True)
    f.close()
    #with open("temp_" + "spla_" + data_name + ".txt") as f:
    #     data = f.read()
    #output_dict = ast.literal_eval(data)
    #df = pd.DataFrame(output_dict)
    #df.to_csv('spla_' + str(data_name) + '.csv', index=False)
