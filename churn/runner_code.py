import pandas as pd
import subprocess
import ast
datasets = ["churn"]
options = 'all'
boundary = " 1 "
bic_enable = "1"
#cs_run(data_name, feature, inv, norm, boundary, bic_enable):
for data_name in datasets:
    if options == 'cs':
        f = open("temp_" + "all_cs_" + data_name + ".txt", "w")
        subprocess.run(["python3 ../cs_run.py " + data_name + " 0 " + " 0 " + " 0 " + boundary + bic_enable], stderr=f, stdout=f, shell=True)
        f.close()
        with open("temp_" + "all_cs_" + data_name + ".txt") as f:
            data = f.read()
        output_dict = ast.literal_eval(data)
        df = pd.DataFrame.from_dict(output_dict, orient='index')
        df = df.transpose()
        df.to_csv('cs_' + str(data_name) + '.csv', index=False)

    if options == 'cs_inv':
        f = open("temp_" + "all_cs_inv_" + data_name + ".txt", "w")
        subprocess.run(["python3 ../cs_run.py " + data_name + " 0 " + " 1 "+ " 0 " + boundary + bic_enable], stderr=f, stdout=f, shell=True)
        f.close()
        with open("temp_" + "all_cs_inv_" + data_name + ".txt") as f:
            data = f.read()
        output_dict = ast.literal_eval(data)
        df = pd.DataFrame.from_dict(output_dict, orient='index')
        df = df.transpose()
        df.to_csv('cs_inv_' + str(data_name) + '.csv', index=False)

    elif options == 'cs_rbf':
        f = open("temp_" + "all_cs_rbf_" + data_name + ".txt", "w")
        subprocess.run(["python3 ../cs_run.py " + data_name + " 1 " + " 0 "+ " 0 " + boundary + bic_enable], stderr=f, stdout=f, shell=True)
        f.close()
        with open("temp_" + "all_cs_rbf_" + data_name + ".txt") as f:
            data = f.read()
        output_dict = ast.literal_eval(data)
        df = pd.DataFrame.from_dict(output_dict, orient='index')
        df = df.transpose()
        df.to_csv('cs_rbf_' + str(data_name) + '.csv', index=False)

    elif options == 'cs_rbf_inv':
        f = open("temp_" + "all_cs_rbf_inv_" + data_name + ".txt", "w")
        subprocess.run(["python3 ../cs_run.py " + data_name + " 1 " + " 1 "+ " 0 " + boundary + bic_enable], stderr=f, stdout=f, shell=True)
        f.close()
        with open("temp_" + "all_cs_rbf_inv_" + data_name + ".txt") as f:
            data = f.read()
        output_dict = ast.literal_eval(data)
        df = pd.DataFrame.from_dict(output_dict, orient='index')
        df = df.transpose()
        df.to_csv('cs_rbf_inv_' + str(data_name) + '.csv', index=False)

    elif options == 'cs_rbf_norm':
        f = open("temp_" + "all_cs_rbf_norm_" + data_name + ".txt", "w")
        subprocess.run(["python3 ../cs_run.py " + data_name + " 1 " + " 0 "+ " 1 " + boundary + bic_enable], stderr=f, stdout=f, shell=True)
        f.close()
        with open("temp_" + "all_cs_rbf_norm_" + data_name + ".txt") as f:
            data = f.read()
        output_dict = ast.literal_eval(data)
        df = pd.DataFrame.from_dict(output_dict, orient='index')
        df = df.transpose()
        df.to_csv('cs_rbf_norm_' + str(data_name) + '.csv', index=False)

    elif options == 'cs_rbf_norm_inv':
        f = open("temp_" + "all_cs_rbf_norm_inv_" + data_name + ".txt", "w")
        subprocess.run(["python3 ../cs_run.py " + data_name + " 1 " + " 1 "+ " 1 " + boundary + bic_enable], stderr=f, stdout=f, shell=True)
        f.close()
        with open("temp_" + "all_cs_rbf_norm_inv_" + data_name + ".txt") as f:
            data = f.read()
        output_dict = ast.literal_eval(data)
        df = pd.DataFrame.from_dict(output_dict, orient='index')
        df = df.transpose()
        df.to_csv('cs_rbf_norm_inv_' + str(data_name) + '.csv', index=False)

    elif options == 'cs_qua':
        f = open("temp_" + "all_cs_qua_" + data_name + ".txt", "w")
        subprocess.run(["python3 ../cs_run.py " + data_name + " 2 " + " 0 "+ " 0 " + boundary + bic_enable], stderr=f, stdout=f, shell=True)
        f.close()
        with open("temp_" + "all_cs_qua_" + data_name + ".txt") as f:
            data = f.read()
        output_dict = ast.literal_eval(data)
        df = pd.DataFrame.from_dict(output_dict, orient='index')
        df = df.transpose()
        df.to_csv('cs_qua_' + str(data_name) + '.csv', index=False)

    elif options == 'cs_qua_inv':
        f = open("temp_" + "all_cs_qua_inv_" + data_name + ".txt", "w")
        subprocess.run(["python3 ../cs_run.py " + data_name + " 2 " + " 1 " + " 0 " + boundary + bic_enable], stderr=f, stdout=f, shell=True)
        f.close()
        with open("temp_" + "all_cs_qua_inv_" + data_name + ".txt") as f:
            data = f.read()
        output_dict = ast.literal_eval(data)
        df = pd.DataFrame.from_dict(output_dict, orient='index')
        df = df.transpose()
        df.to_csv('cs_qua_inv_' + str(data_name) + '.csv', index=False)

    elif options == 'cs_qua_norm':
        f = open("temp_" + "all_cs_qua_norm_" + data_name + ".txt", "w")
        subprocess.run(["python3 ../cs_run.py " + data_name + " 2 " + " 0 " + " 1 " + boundary + bic_enable], stderr=f, stdout=f, shell=True)
        f.close()
        with open("temp_" + "all_cs_qua_norm_" + data_name + ".txt") as f:
            data = f.read()
        output_dict = ast.literal_eval(data)
        df = pd.DataFrame.from_dict(output_dict, orient='index')
        df = df.transpose()
        df.to_csv('cs_qua_norm_' + str(data_name) + '.csv', index=False)

    elif options == 'cs_qua_norm_inv':
        f = open("temp_" + "all_cs_qua_norm_inv_" + data_name + ".txt", "w")
        subprocess.run(["python3 ../cs_run.py " + data_name + " 2 " + " 1 "+ " 1 " + boundary + bic_enable], stderr=f, stdout=f, shell=True)
        f.close()
        with open("temp_" + "all_cs_qua_norm_inv_" + data_name + ".txt") as f:
            data = f.read()
        output_dict = ast.literal_eval(data)
        df = pd.DataFrame.from_dict(output_dict, orient='index')
        df = df.transpose()
        df.to_csv('cs_qua_norm_inv_' + str(data_name) + '.csv', index=False)

    elif options == "all":
        f = open("temp_" + "all_cs_" + data_name + ".txt", "w")
        subprocess.run(["python3 ../cs_run.py " + data_name + " 0 " + " 0 "+ " 0 " + boundary + bic_enable], stderr=f, stdout=f, shell=True)
        f.close()

        f = open("temp_" + "all_cs_inv_" + data_name + ".txt", "w")
        subprocess.run(["python3 ../cs_run.py " + data_name + " 0 " + " 1 "+ " 0 " + boundary + bic_enable], stderr=f, stdout=f, shell=True)
        f.close()

        f = open("temp_" + "all_cs_rbf_" + data_name + ".txt", "w")
        subprocess.run(["python3 ../cs_run.py " + data_name + " 1 " + " 0 " + " 0 " + boundary + bic_enable], stderr=f, stdout=f, shell=True)
        f.close()

        f = open("temp_" + "all_cs_rbf_inv_" + data_name + ".txt", "w")
        subprocess.run(["python3 ../cs_run.py " + data_name + " 1 " + " 1 " + " 0 " + boundary + bic_enable], stderr=f, stdout=f, shell=True)
        f.close()

        f = open("temp_" + "all_cs_rbf_norm_" + data_name + ".txt", "w")
        subprocess.run(["python3 ../cs_run.py " + data_name + " 1 " + " 0 "+ " 1 " + boundary + bic_enable], stderr=f, stdout=f, shell=True)
        f.close()

        f = open("temp_" + "all_cs_rbf_norm_inv_" + data_name + ".txt", "w")
        subprocess.run(["python3 ../cs_run.py " + data_name + " 1 " + " 1 "+ " 1 " + boundary + bic_enable], stderr=f, stdout=f, shell=True)
        f.close()

        f = open("temp_" + "all_cs_qua_" + data_name + ".txt", "w")
        subprocess.run(["python3 ../cs_run.py " + data_name + " 2 " + " 0 "+ " 0 " + boundary + bic_enable], stderr=f, stdout=f, shell=True)
        f.close()

        f = open("temp_" + "all_cs_qua_inv_" + data_name + ".txt", "w")
        subprocess.run(["python3 ../cs_run.py " + data_name + " 2 " + " 1 " + " 0 " + boundary + bic_enable], stderr=f, stdout=f, shell=True)
        f.close()

        f = open("temp_" + "all_cs_qua_norm_" + data_name + ".txt", "w")
        subprocess.run(["python3 ../cs_run.py " + data_name + " 2 " + " 0 " + " 1 " + boundary + bic_enable], stderr=f, stdout=f, shell=True)
        f.close()

        f = open("temp_" + "all_cs_qua_norm_inv_" + data_name + ".txt", "w")
        subprocess.run(["python3 ../cs_run.py " + data_name + " 2 " + " 1 " + " 1 " + boundary + bic_enable], stderr=f, stdout=f, shell=True)
        f.close()

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
        df = df.transpose()
        df.to_csv('cs_qua_norm_inv_' + str(data_name) + '.csv', index=False)
        output.append(output_dict)

        df = pd.DataFrame.from_records(output)
        df.to_csv('all_' + str(data_name) + '.csv', index=False)
