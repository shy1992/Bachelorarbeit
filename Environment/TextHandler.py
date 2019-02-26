from Config import *
from Model import *
from DatasetOperations import *

import numpy as np
import tensorflow as tf
import cv2
import os


# Wandle das Dataset in eine Repräsentation um
def createEmbeddingDataset(epoch, check_point_file = "model/modelHandConv.ckpt"):

    # Öffne Dateien und erfasse die relevanten Knotenpunkte 
    dset_file = h5py.File("Dataset/datasetOrdered.hdf5", "r")
    x_dataset = dset_file["training"]
    graph = tf.Graph()
    with graph.as_default():
        x = tf.placeholder("float", shape=[None, input_shape[0],input_shape[1],1], name="input")
        g_end, z, optimizer, cost,z_mean,z_log_sigma_sq = createGraph(x, False, True)

        # Lade das aktuelle Modell
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, check_point_file)

            # Setze den Pointer und erstelle eine Matrix die die Repräsentationen speuchern soll
            pointer = 0
            embeddings = np.zeros((MAX_NUMBERS_EMBEDDING, n_z))
            # Iteriere durch den Datenbestand
            while(True):
                batch_xs, pointer = getBatch(x_dataset, batch_size, pointer)
                batch_xs = batch_xs.reshape(batch_size,input_shape[0],input_shape[1],1)
                batch_xs = batch_xs/255
                
                z_baseline  = sess.run((z), feed_dict={x:batch_xs})

                # Sollte die Menge an Datenpunkten eine vordefinierte Menge überschreiten breche den Vorgang ab 
                if(pointer< MAX_NUMBERS_EMBEDDING):
                    embeddings[pointer-batch_size:pointer,:] = z_baseline
                else:
                    embeddings[pointer-batch_size:MAX_NUMBERS_EMBEDDING,:] = z_baseline
                    break

            # Schreibe den Datensatz
            PATH = os.getcwd()
            LOG_DIR = PATH+ '/TextSaves'
            feature_path = os.path.join(LOG_DIR, 'embeddings_{}'.format(epoch))
            np.save(feature_path, embeddings)
    dset_file.close()
    


# Hänge den neuen Wert an den bestehenden Textdokument an / oder erstelle neues Textdokument falls nötig
def appendCost(epoch, cost):
    PATH = os.getcwd()
    LOG_DIR = PATH+ '/TextSaves'
    cost_path = os.path.join(LOG_DIR, 'cost.txt')
    if epoch == 0:
        with open(cost_path, "w") as f:
            f.write("{};{}\n".format(epoch,cost))
    else:
        with open(cost_path, "a") as f:
            f.write("{};{}\n".format(epoch,cost))
            
