import numpy as np

def read_data():
    data = np.load('/scratch365/lwei5/FEASAI_data/Train/0001.npz')
    for key in data:
        print(key)
        print(data[key].shape)

read_data()