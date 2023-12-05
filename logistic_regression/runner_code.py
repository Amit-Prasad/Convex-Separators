import pandas as pd
import subprocess
import ast
datasets = ["churn", "telco_churn", "santander_sub", "covtype_bin_sub", "spambase", "shoppers", "diabetes", "breast_cancer", "ionosphere", "philippine"]

for data_name in datasets:
    f = open("temp_" + "lr_" + data_name + ".txt", "w")
    subprocess.run(["python3 logistic_regression.py " + data_name], stderr=f, stdout=f, shell=True)
    f.close()
