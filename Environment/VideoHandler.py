from Config import *
from Model import *
from DatasetOperations import *

import numpy as np
import tensorflow as tf
import cv2
import os


# Dient als Schnittstelle falls noch andere Videos erstellt werden sollen
def createVideo(n_z, epoch, gaussianPlot=True, check_point_file = "model/modelHandConv.ckpt"):
    createStutterVideo(n_z, epoch, check_point_file, stuttering)



# Video mit niedrigeren Frames beinhaltet groundtruth - groundtruth prediction - training set - interpolationsVideo
def createStutterVideo(n_z, epoch, check_point_file, cutOut):
    # Erstelle einen neuen TensorFlow Graphen und setze ihn als default
    graph = tf.Graph()
    with graph.as_default():

        # Erfasse relevante Knotenpunkte innerhalb des Graphen
        x = tf.placeholder("float", shape=[None, input_shape[0],input_shape[1],1], name="input")
        g_end, z, optimizer, cost,z_mean,z_log_sigma_sq = createGraph(x, False, True)

        # Lade das vollständige Video
        path = os.path.join("Dataset/OriginalVideo.avi")
        cap = cv2.VideoCapture(path)
        title = "Morphed/StutterVideo_z{}_e{}.avi".format(n_z, epoch)
        out = cv2.VideoWriter(title, cv2.VideoWriter_fourcc('M','J','P','G'), 30, (64*2,64*2))

        # Der Counter für den reduzierten Datensatz um den zu tracken und das richtige element zu erfassen
        increment = 0
        dset_f = h5py.File("Dataset/1to"+str(cutOut)+"OrderedDataset.hdf5", "r")
        orderedDataset = dset_f["training"]

        saver = tf.train.Saver()
        with tf.Session() as sess:
            # Lade das aktuelle Modell
            saver.restore(sess, check_point_file)

            # Iteriere durch das Video
            while(True):
                # Erhalte den neuen Eintrag
                loop, data = cap.read()

                if(loop):
                    # Wandle das OriginalVideoBild in ein Graubild um
                    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
                    data = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)

                    # Wandle das Bild in eine Repräsentation um
                    z_m, z_sigma  = sess.run((z_mean, z_log_sigma_sq), feed_dict={x:data.reshape(1,input_shape[0], input_shape[1], 1)/255})
                    resultImage = sess.run((g_end), feed_dict={z: z_m})
                    resultImage = resultImage*255
                    resultImage = resultImage.reshape((64,64))

                 
                    # Erstelle ein Canvas und setze die obere Zeile mit den berechneten Daten
                    canvas = np.zeros((input_shape[0]*2, input_shape[1]*2))
                    canvas[0:input_shape[1], 0:input_shape[1]] = data
                    canvas[0:input_shape[1],input_shape[1]:] = resultImage


                    ## Die zweite Reihe
                    # Interpolationsschritt
                    stutterIndex = int(increment/cutOut)
                    stutterRest = increment%cutOut / cutOut
                    stutterImage = orderedDataset[stutterIndex]
                    try:
                        stutterImageNext = orderedDataset[stutterIndex+1]
                    except:
                        stutterImageNext = orderedDataset[stutterIndex]
                    # Erhalte die beiden Repräsentationen von Bild und Bild+1 
                    z_m, z_sigma  = sess.run((z_mean, z_log_sigma_sq), feed_dict={x:stutterImage.reshape(1,input_shape[0], input_shape[1], 1)/255})
                    z_mNext, z_sigma  = sess.run((z_mean, z_log_sigma_sq), feed_dict={x:stutterImageNext.reshape(1,input_shape[0], input_shape[1], 1)/255})
                    diffVector = z_mNext-z_m
                    z_interpolation = z_m + stutterRest * diffVector

                    # Generiere das Bild für den Interpolationsschritt
                    resultImage = sess.run((g_end), feed_dict={z: z_interpolation})
                    resultImage = resultImage*255
                    resultImage = resultImage.reshape((64,64))

                    # Setze die Daten in das Canvas ein 
                    canvas[input_shape[1]:, 0:input_shape[1]] = stutterImage
                    canvas[input_shape[1]:, input_shape[1]:] = resultImage          

                    # Konvertiere das Canvas in ein gültiges Format zum abspeichern
                    canvas = np.array(canvas, dtype=np.uint8)                  
                    canvas = canvas.reshape((64*2,64*2,1))
                    grayscaled = np.zeros((canvas.shape[0], canvas.shape[1],3))
                    grayscaled[:, :,0] = canvas[:,:,0]
                    grayscaled[:, :,1] = canvas[:,:,0]
                    grayscaled[:, :,2] = canvas[:,:,0]
                    grayscaled = grayscaled.astype(np.uint8)
                    writeFrame = cv2.cvtColor(grayscaled, cv2.COLOR_RGB2BGR)
                    
            
                    # Schreibe den Frame
                    out.write(writeFrame)
                    increment +=1
                   
                else:  
                    break
    # Schließe sämtliche offenen Dateien
    dset_f.close()               
    cap.release()
    out.release()
    cv2.destroyAllWindows()


# Wenn es sich bei dem Aufruf dieses Skripts um das Hauptprogramm handelt wird ein Video basierend auf dem aktuellen Model "modelHandConv.ckpt" erstellt
if __name__ == "__main__":
    check_point_file = "model/modelHandConv.ckpt"
    createStutterVideo(25, 20, check_point_file, 2)    

