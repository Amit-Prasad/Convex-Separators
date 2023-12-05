#cleaveland hear disease dataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def get_dataset(name):
    if name == 'circle':
        data = pd.read_csv('data_dist_circle.csv', header=None).values
        data_x = data[:, 0:-1]
        data_y = data[:, -1]
        data_y = np.where(data_y == -1, 0, 1)
        data_x, data_x_test, data_y, data_y_test = train_test_split(data_x, data_y, test_size=0.30, random_state=1)
        return data_x, data_y, data_x_test, data_y_test

    if name == 'triangle':
        data = pd.read_csv('triangle.csv', header=None).values
        data_x = data[:, 0:-1]
        data_y = data[:, -1]
        data_y = np.where(data_y == -1, 0, 1)
        data_x, data_x_test, data_y, data_y_test = train_test_split(data_x, data_y, test_size=0.30, random_state=1)
        return data_x, data_y, data_x_test, data_y_test

    if name == 'square':
        data = pd.read_csv('square.csv', header=None).values
        data_x = data[:, 0:-1]
        data_y = data[:, -1]
        data_y = np.where(data_y == -1, 0, 1)
        data_x, data_x_test, data_y, data_y_test = train_test_split(data_x, data_y, test_size=0.30, random_state=1)
        return data_x, data_y, data_x_test, data_y_test

    if name == 'pentagon':
        data = pd.read_csv('pentagon.csv', header=None).values
        data_x = data[:, 0:-1]
        data_y = data[:, -1]
        data_y = np.where(data_y == -1, 0, 1)
        data_x, data_x_test, data_y, data_y_test = train_test_split(data_x, data_y, test_size=0.30, random_state=1)
        return data_x, data_y, data_x_test, data_y_test

    if name == 'hexagon':
        data = pd.read_csv('hexagon.csv', header=None).values
        data_x = data[:, 0:-1]
        data_y = data[:, -1]
        data_y = np.where(data_y == -1, 0, 1)
        data_x, data_x_test, data_y, data_y_test = train_test_split(data_x, data_y, test_size=0.30, random_state=1)
        return data_x, data_y, data_x_test, data_y_test

    if name == 'diamond':
        data = pd.read_csv('diamond.csv', header=None).values
        data_x = data[:, 0:-1]
        data_y = data[:, -1]
        data_y = np.where(data_y == -1, 0, 1)
        data_x, data_x_test, data_y, data_y_test = train_test_split(data_x, data_y, test_size=0.30, random_state=1)
        return data_x, data_y, data_x_test, data_y_test

    if name == 'ellipse':
        data = pd.read_csv('ellipse.csv', header=None).values
        data_x = data[:, 0:-1]
        data_y = data[:, -1]
        data_y = np.where(data_y == -1, 0, 1)
        data_x, data_x_test, data_y, data_y_test = train_test_split(data_x, data_y, test_size=0.30, random_state=1)
        return data_x, data_y, data_x_test, data_y_test

    if name == 'ellipsoid':
        data = pd.read_csv('ellipsoid.csv', header=None).values
        data_x = data[:, 0:-1]
        data_y = data[:, -1]
        data_y = np.where(data_y == -1, 0, 1)
        data_x, data_x_test, data_y, data_y_test = train_test_split(data_x, data_y, test_size=0.30, random_state=1)
        return data_x, data_y, data_x_test, data_y_test

    if name == 'ellipsoid_7D':
        data = pd.read_csv('ellipsoid_7D.csv', header=None).values
        data_x = data[:, 0:-1]
        data_y = data[:, -1]
        data_y = np.where(data_y == -1, 0, 1)
        data_x, data_x_test, data_y, data_y_test = train_test_split(data_x, data_y, test_size=0.30, random_state=1)
        return data_x, data_y, data_x_test, data_y_test

    if name == 'sphere_7D':
        data = pd.read_csv('sphere_7D.csv', header=None).values
        data_x = data[:, 0:-1]
        data_y = data[:, -1]
        data_y = np.where(data_y == -1, 0, 1)
        data_x, data_x_test, data_y, data_y_test = train_test_split(data_x, data_y, test_size=0.30, random_state=1)
        return data_x, data_y, data_x_test, data_y_test

