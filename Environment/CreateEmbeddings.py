from Config import *

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


import h5py
import math

tfd = tf.contrib.distributions



dset_file = h5py.File("Dataset/datasetOrdered.hdf5", "r")
x_dataset = dset_file["training"]
dset_shape = dset_file["training"].shape
input_shape = dset_file["training"][0].shape
print("Datasetshape:", dset_shape)
print("Input shape:", input_shape)



PATH = os.getcwd()
LOG_DIR = PATH+ '/embeddingModel'
print("Save Features")
feature_path = os.path.join(LOG_DIR, 'features')
"""
np.save(feature_path, embeddings)
print("saved...")
"""

embeddings = 0 ## Save RAM

tf.reset_default_graph()
"""
print("Create Meta")
metadata_file = open(os.path.join(LOG_DIR, 'metadata.tsv'), 'w')
metadata_file.write('Class\tName\n')

for j in range(MAX_NUMBERS):
    metadata_file.write('{0}\t{}\n'.format(j,"Hand"))
metadata_file.close()
"""

"""
print("Create Sprite")
## Generate Sprite
#Note: We currently support sprites up to 8192px X 8192px.
##  8192/64 = 128
sprite = np.zeros((8192,8192,1), dtype=np.float32)
for x in range(128):
    for y in range(128):
        catch = x_dataset[x*128+y]
        catch = catch.reshape((64,64,1))
        sprite[x*64:(x+1)*64, y*64:(y+1)*64] = catch
        
cv2.imwrite(os.path.join(LOG_DIR, 'sprite.png'), sprite)
"""

print("Create Logs")
## Create Logs
with tf.Session() as sess: 

    
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    
    for iterate in range(79):
        path = PATH+'/TextSaves/embeddings_'+str(iterate)+".npy"
        #print(path)
        embedding_var = tf.Variable(np.load(path), name="Z_Epoch{}".format(iterate))
        embedding.tensor_name = embedding_var.name

    
    
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(LOG_DIR, 'checkpoint.ckpt'))



    # Use the same LOG_DIR where you stored your checkpoint.
    summary_writer = tf.summary.FileWriter(LOG_DIR)

    # The next line writes a projector_config.pbtxt in the LOG_DIR. TensorBoard will
    # read this file during startup.
    projector.visualize_embeddings(summary_writer, config)

    embedding.sprite.image_path = os.path.join(LOG_DIR, 'sprite.png')
    # Specify the width and height of a single thumbnail.
    embedding.sprite.single_image_dim.extend([64, 64])

    projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)



    

    #embedding.metadata_path = "metadata.tsv"#os.path.join(LOG_DIR, 'metadata.tsv')

    # Use the same LOG_DIR where you stored your checkpoint.
    summary_writer = tf.summary.FileWriter(LOG_DIR)

    # The next line writes a projector_config.pbtxt in the LOG_DIR. TensorBoard will
    # read this file during startup.
    projector.visualize_embeddings(summary_writer, config)

    embedding.sprite.image_path = os.path.join(LOG_DIR, 'sprite.png')
    # Specify the width and height of a single thumbnail.
    embedding.sprite.single_image_dim.extend([64, 64])

    projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)







"""


metadata_file = open(os.path.join(LOG_DIR, 'metadata_4_classes.tsv'), 'w')
metadata_file.write('Class\tName\n')






tfd = tf.contrib.distributions

path = os.path.join("outpy.avi")
cap = cv2.VideoCapture(path)

loop, data = cap.read()

data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
data = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)
x_dataset = np.array([data], dtype=np.float32)

with h5py.File("datasetOrdered.hdf5", "a") as f:
    f.create_dataset("training", data=x_dataset , maxshape=(None,64,64),dtype=np.float32, chunks=True)
    
    
    while(True):
        loop, data = cap.read()

        if(loop):
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
            data = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)
            x_dataset = np.array([data], dtype=np.float32)
            
            f["training"].resize((f["training"].shape[0]+1, 64,64),)
            
            f["training"][-1] = x_dataset
            
            f.flush()
            
            #print(data.shape)
            #x_dataset = np.append(x_dataset, np.array([data]), axis=0)
        else:
            print("Shuffling started...")
            #shuffle(f["training"])
            break

    print("Trainingsdataset shape:", f["training"].shape)


#plt.imshow(x_dataset[0], cmap=plt.get_cmap('gray'))
plt.figure(figsize=(15, 4))
with h5py.File("dataset.hdf5", "r") as f:
    for i in range(12):
        randVal = randint(0,f["training"].shape[0]-1)
        plt.subplot(2, 6, i+1)
        plt.imshow(f["training"][randVal],cmap=plt.get_cmap('gray'))
plt.show()










def getRandomImages(d_set,anzahl):
    bla = randint(0, d_set.shape[0]-1)
    x_sample = np.array([d_set[bla].reshape(input_shape[0], input_shape[1], 1)], dtype=np.float32)
    
    for k in range(anzahl-1):
        bla = randint(0, d_set.shape[0]-1)
        x_sample = np.append(x_sample,  np.array([d_set[bla].reshape(input_shape[0], input_shape[1], 1)]), axis = 0)
    return x_sample      

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
    

randSample = getRandomImages(x_dataset, 5)
plt.figure(figsize=(15,6))
for haha in range(5):
    plt.subplot(1, 5, haha+1)
    plt.imshow(randSample[haha].reshape(input_shape[0],input_shape[1]),cmap=plt.get_cmap('gray'))
    plt.title("Example")
    num = 0

plt.show()
plt.close()
"""
