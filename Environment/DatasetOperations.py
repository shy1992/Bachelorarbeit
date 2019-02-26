from Config import *

import numpy as np
from random import randint
import h5py
import matplotlib.pyplot as plt


# Lade das Datenset von einem H5PY Dateiformat
dset_file = h5py.File(DATASET_NAME, "r")
x_dataset = dset_file[DATASET_DB_NAME]
dset_shape = dset_file[DATASET_DB_NAME].shape
input_shape = dset_file[DATASET_DB_NAME][0].shape
print("Datasetshape:", dset_shape)
print("Image shape:", input_shape)


# Erhalte eine zufällige Auswahlt an Bildern aus dem Datenbestand
def getRandomImages(d_set, anzahl):
    bla = randint(0, d_set.shape[0]-1)
    x_sample = np.array([d_set[bla].reshape(input_shape[0], input_shape[1], 1)], dtype=np.float32)

    for k in range(anzahl-1):
        bla = randint(0, d_set.shape[0]-1)
        x_sample = np.append(x_sample,  np.array([d_set[bla].reshape(input_shape[0], input_shape[1], 1)]), axis = 0)
    return x_sample      


# Erhalte eine Minibatcheinheit je nach Pointer
def getBatch(x_dataset, batchSize, pointer):
    shape = x_dataset.shape[0]
    if pointer+batch_size < shape:
        tmp = pointer
        pointer += batchSize
        return (x_dataset[tmp:pointer],pointer)
    else:
        tmp = 0
        pointer += batchSize
        return (x_dataset[tmp:batchSize], tmp)


# Zeige zu Beginn des Trainings ein paar Beispielsbilder
if INSPECT_DATASET:
    randSample = getRandomImages(x_dataset, 5)
    plt.figure(figsize=(15,6))
    for haha in range(5):
        plt.subplot(1, 5, haha+1)
        plt.imshow(randSample[haha].reshape(input_shape[0],input_shape[1]),cmap=plt.get_cmap('gray'))
        plt.title("Example")
    
    plt.show()

