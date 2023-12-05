import pandas as pd
import ast

data_name = "churn"
output = []
with open("temp_" + "all_cs_" + data_name + ".txt") as f:
    data = f.read()
output_dict = ast.literal_eval(data)
df = pd.DataFrame.from_dict(output_dict, orient='index')
df = df.transpose()
df.to_csv('cs_' + str(data_name) + '.csv', index=False)
output.append(output_dict)

with open("temp_" + "all_cs_inv_" + data_name + ".txt") as f:
    data = f.read()
output_dict = ast.literal_eval(data)
df = pd.DataFrame.from_dict(output_dict, orient='index')
df = df.transpose()
df.to_csv('cs_inv_' + str(data_name) + '.csv', index=False)
output.append(output_dict)

with open("temp_" + "all_cs_rbf_" + data_name + ".txt") as f:
    data = f.read()
output_dict = ast.literal_eval(data)
df = pd.DataFrame.from_dict(output_dict, orient='index')
df = df.transpose()
df.to_csv('cs_rbf_' + str(data_name) + '.csv', index=False)
output.append(output_dict)

with open("temp_" + "all_cs_rbf_inv_" + data_name + ".txt") as f:
    data = f.read()
output_dict = ast.literal_eval(data)
df = pd.DataFrame.from_dict(output_dict, orient='index')
df = df.transpose()
df.to_csv('cs_rbf_inv_' + str(data_name) + '.csv', index=False)
output.append(output_dict)

with open("temp_" + "all_cs_rbf_norm_" + data_name + ".txt") as f:   
     data = f.read()
output_dict = ast.literal_eval(data)
df = pd.DataFrame.from_dict(output_dict, orient='index')
df = df.transpose()
df.to_csv('cs_rbf_norm_' + str(data_name) + '.csv', index=False)
output.append(output_dict)

with open("temp_" + "all_cs_rbf_norm_inv_" + data_name + ".txt") as f:
    data = f.read()
output_dict = ast.literal_eval(data)
df = pd.DataFrame.from_dict(output_dict, orient='index')
df = df.transpose()
df.to_csv('cs_rbf_norm_inv_' + str(data_name) + '.csv', index=False)
output.append(output_dict)

with open("temp_" + "all_cs_qua_" + data_name + ".txt") as f:
   data = f.read()
output_dict = ast.literal_eval(data)
df = pd.DataFrame.from_dict(output_dict, orient='index')
df = df.transpose()
df.to_csv('cs_qua_' + str(data_name) + '.csv', index=False)
output.append(output_dict)

with open("temp_" + "all_cs_qua_inv_" + data_name + ".txt") as f:
   data = f.read()
output_dict = ast.literal_eval(data)
df = pd.DataFrame.from_dict(output_dict, orient='index')
df = df.transpose()
df.to_csv('cs_qua_inv_' + str(data_name) + '.csv', index=False)
output.append(output_dict)

with open("temp_" + "all_cs_qua_norm_" + data_name + ".txt") as f:
   data = f.read()
output_dict = ast.literal_eval(data)
df = pd.DataFrame.from_dict(output_dict, orient='index')
df = df.transpose()
df.to_csv('cs_qua_norm_' + str(data_name) + '.csv', index=False)
output.append(output_dict)

with open("temp_" + "all_cs_qua_norm_inv_" + data_name + ".txt") as f:
   data = f.read()
output_dict = ast.literal_eval(data)
df = pd.DataFrame.from_dict(output_dict, orient='index')
df.to_csv('cs_qua_norm_inv_' + str(data_name) + '.csv', index=False)
output.append(output_dict)
df = pd.DataFrame.from_records(output)
df.to_csv('all_' + str(data_name) + '.csv', index=False)
