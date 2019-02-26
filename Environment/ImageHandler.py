from Config import *
from Model import *
from DatasetOperations import *

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import os



# Erstellt den latenten Plot
def createLatentplot(epoch, check_point_file = "model/modelHandConv.ckpt"):
    # Erstelle neuen Graph
    graph = tf.Graph()
    with graph.as_default():
        x_tensor = tf.placeholder("float", shape=[None, input_shape[0],input_shape[1],1], name="input")
        g_end, z, optimizer, cost,z_mean,z_log_sigma_sq = createGraph(x_tensor, False, True)

        # Erstelle neue Session und lade das Modell
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, check_point_file)

            # Erstellt jeweils 12 verschiedene Versionen des latenten Plots
            for wahl in range(12):
                z_baseline  = sess.run((z), feed_dict={x_tensor: x_dataset[wahl*100].reshape(1,input_shape[0], input_shape[1], 1)/255})
                resultImage = sess.run((g_end), feed_dict={z: z_baseline})

                plt.subplot(2,1,1)
                resultImage = resultImage.reshape(input_shape[0], input_shape[1])*255
                plt.imshow(resultImage, cmap=plt.get_cmap('gray'))
                
                
                plt.subplot(2,1,2)
                
                ## Plotten der Latenten Eigenschaften
                nx = 15
                ny = n_z
                x_values = np.linspace(-4, 4, nx)
                canvas = np.empty((input_shape[0]*ny, input_shape[1]*nx))
                for i in range(ny):
                    z_latent = z_baseline

                    # Erstelle iene Matrix mit veränderten Werten in der Zeile i
                    z_latent = np.ones((nx,ny))*z_latent
                    z_latent[:,i] += x_values.transpose()

                    # Generiere die entsprechenden Bilder
                    resultImage = sess.run(g_end, feed_dict={z: z_latent})

                    # Schreibe jedes Bild an die richtige Stelle
                    for canvasRow in range(nx):
                        canvas[(i)*input_shape[0]:(i+1)*input_shape[1], canvasRow*input_shape[0]:(canvasRow+1)*input_shape[1]] = resultImage[canvasRow].reshape(input_shape[0], input_shape[1])

                # Formatiere den Plot
                plt.figure(figsize=(15, 15))        
                plt.imshow(canvas, origin="upper", vmin=0, vmax=1,interpolation='none',cmap=plt.get_cmap('gray'))
                plt.tight_layout()

                # Speichern der Grafik
                file_name = "latentplot_{}_z{}_e{}".format(wahl, n_z, epoch)
                p = os.path.join("GeneratedImage", file_name)
                plt.savefig(p)
                plt.close()




# Erstellt ein ReconstructionImage zwischen der Eingabe und dem wiederhergestellten Bild
def createReconstructionImage(epoch, check_point_file = "model/modelHandConv.ckpt"):
    # Erstelle neuen Graph
    graph = tf.Graph()
    with graph.as_default():
        x = tf.placeholder("float", shape=[None, input_shape[0],input_shape[1],1], name="input")
        g_end, z, optimizer, cost, z_mean, z_log_sigma_sq = createGraph(x, False, True)

        # Neue Session starten und Modell laden
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, check_point_file)

            # Lade 5 Random Bilder und benutze diese Bilder als Eingabe für den VAE
            randSample = getRandomImages(x_dataset, 5)/255
            resultPlot = sess.run((g_end), feed_dict={x: randSample})

            # Erstelle den Plot mit der ersten Zeile Ground truth und die zweite Zeile die Vorhersagen
            plt.figure(figsize=(15,6))
            for haha in range(5):
                plt.subplot(2, 5, haha+1)
                plt.imshow(randSample[haha].reshape(input_shape[0],input_shape[1])*255,cmap=plt.get_cmap('gray'))
                plt.title("Input epoch:{}".format(epoch))
            for haha in range(5):
                plt.subplot(2, 5, haha+6)
                plt.imshow(resultPlot[haha].reshape(input_shape[0],input_shape[1])*255,cmap=plt.get_cmap('gray'))
                plt.title("Result epoch:{}".format(epoch))

            # Speichere die Grafik
            file_name = "reconstruction_z{}_e{}".format(n_z, epoch)
            p = os.path.join("GeneratedImage", file_name)
            plt.savefig(p)
            plt.close()

