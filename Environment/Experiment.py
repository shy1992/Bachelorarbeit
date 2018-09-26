from Config import *
from DatasetOperations import *
from Model import *
from ImageHandler import *
from TextHandler import *
from VideoHandler import *




import tensorflow as tf
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import h5py
import math
from random import randint
import datetime

import time
import cv2



g_end, z, optimizer, cost = createGraph(x)


## Train Hand Autoencoder Model
runs =  201
pointer = 0
num = 0

offset = 0

print("Starte Training von {} bis {}".format(offset, offset+runs))

init = tf.global_variables_initializer()
saver = tf.train.Saver()

minibatch_epoch = int(dset_shape[0]/batch_size)

print("Anzahl der Minibatches:", minibatch_epoch)

with tf.Session() as sess:
    sess.run(init)

    if offset != 0:
        saver.restore(sess, "model/modelHandConv.ckpt")
        print("Model restored")
    
    batch_xs, pointer = getBatch(x_dataset, batch_size, pointer)
    
    batch_xs = batch_xs.reshape(batch_size,input_shape[0],input_shape[1],1)

    dd = sess.run([cost], feed_dict={x: batch_xs})
    print('Test run after starting {}'.format(dd))
    

    for epoch in range(offset, runs + offset):
        avg_cost = 0.
        print("Training:", epoch, "at", datetime.datetime.now())
        # Loop over all batches

        for i in range(minibatch_epoch):
            batch_xs, pointer = getBatch(x_dataset, batch_size, pointer)
            batch_xs = batch_xs.reshape(batch_size,input_shape[0],input_shape[1],1)
            
            batch_xs = batch_xs/255
            
            _,d = sess.run((optimizer, cost), feed_dict={x: batch_xs})
            avg_cost += d / minibatch_epoch
        print("Cost in epoch {}: {}".format(epoch, avg_cost))
        appendCost(epoch, avg_cost)
        save_path = saver.save(sess, "model/modelHandConv.ckpt")
        createEmbeddingDataset(epoch)
        
        if epoch % CHECKPOINT_INTERVALL == 0:
            save_path = saver.save(sess, "model/modelHandConv{}.ckpt".format(epoch))
            
        if epoch % LATENT_INTERVALL == 0:
            print("Create Latentplot for epoch ", epoch, "at", datetime.datetime.now())
            createLatentplot(epoch)

        if epoch % RECONSTRUCTION_INTERVALL == 0:
            print("ReconstructionImage for epoch ", epoch, "at", datetime.datetime.now())
            createReconstructionImage(epoch)

        if epoch % VIDEO_INTERVALL == 0 and epoch != 0:
            print("Create Video for epoch ", epoch, "at", datetime.datetime.now())
            createVideo(n_z, epoch)
