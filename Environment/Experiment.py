from Config import *
from DatasetOperations import *
from Model import *
from ImageHandler import *
from TextHandler import *
from VideoHandler import *


import tensorflow as tf
import numpy as np
from random import randint
import datetime



# Erstelle einen Graphen
g_end, z, optimizer, cost = createGraph(x)


## Train Hand Autoencoder Modell
print("Starte Training von {} bis {}".format(OFFSET, OFFSET + RUNS))

# Initialisiere die Variablen
init = tf.global_variables_initializer()
saver = tf.train.Saver()


pointer = 0         # Der Pointer zeigt auf den jeweiligen Batchbestand der aktuell geladen wurde er durchläuft den gesamten Datenbestand in Einzelschritten
minibatch_epoch = int(dset_shape[0]/batch_size) # Anzahl der Minibactheinheiten
print("Anzahl der Minibatches:", minibatch_epoch)


# Initialisiere und lade Modell falls nötig
with tf.Session() as sess:
    sess.run(init)
    if OFFSET != 0:
        saver.restore(sess, "model/modelHandConv.ckpt")
        print("Model restored")

    # Testrun 
    batch_xs, pointer = getBatch(x_dataset, batch_size, pointer)
    batch_xs = batch_xs.reshape(batch_size,input_shape[0],input_shape[1],1)
    dd = sess.run([cost], feed_dict={x: batch_xs})
    print('Test run after starting {}'.format(dd))
    
    # Iteriere durch die einzelnen Epochen
    for epoch in range(OFFSET, RUNS + OFFSET):
        avg_cost = 0.
        print("Training:", epoch, "at", datetime.datetime.now())

        # Iteriere durch den gesamten Datenbestand mit den einzelnen Minibatcheinheiten
        for i in range(minibatch_epoch):
            batch_xs, pointer = getBatch(x_dataset, batch_size, pointer)
            batch_xs = batch_xs.reshape(batch_size,input_shape[0],input_shape[1],1)
            batch_xs = batch_xs/255

            # Berechne die Kosten und führe eine Optimierung durch
            _,d = sess.run((optimizer, cost), feed_dict={x: batch_xs})
            avg_cost += d / minibatch_epoch
        print("Cost in epoch {}: {}".format(epoch, avg_cost))
        # Speichere die Kosten als Textdokument
        appendCost(epoch, avg_cost)
        # Speichere das aktuelle Modell
        save_path = saver.save(sess, "model/modelHandConv.ckpt")
        # Erstelle ein embeddingsDataset
        createEmbeddingDataset(epoch)

        # Speichere das Modell unter einem speziellen namen in vorgegebenen Intervallen
        if epoch % CHECKPOINT_INTERVALL == 0:
            save_path = saver.save(sess, "model/modelHandConv{}.ckpt".format(epoch))

        # Erstelle einen latenten Plot in vorgegebenen Intervallen
        if epoch % LATENT_INTERVALL == 0:
            print("Create Latentplot for epoch ", epoch, "at", datetime.datetime.now())
            createLatentplot(epoch)

        # Erstelle eine Rekonstruktionsimage in vorgegebenen Intervallen
        if epoch % RECONSTRUCTION_INTERVALL == 0:
            print("ReconstructionImage for epoch ", epoch, "at", datetime.datetime.now())
            createReconstructionImage(epoch)
            
        # Erstelle eine Video in vorgegebenen Intervallen
        if epoch % VIDEO_INTERVALL == 0 and epoch != 0:
            print("Create Video for epoch ", epoch, "at", datetime.datetime.now())
            createVideo(n_z, epoch)
