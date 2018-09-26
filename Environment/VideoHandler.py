from Config import *
from Model import *
from DatasetOperations import *

import numpy as np
import tensorflow as tf


import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

import time
import cv2
import os
import sys



def createVideo(n_z, epoch, gaussianPlot=True, check_point_file = "model/modelHandConv.ckpt"):
    if gaussianPlot:
        createGaussianVideo(n_z, epoch, check_point_file)
    else:
        createNormalVideo(n_z, epoch, check_point_file)





def createGaussianVideo(n_z, epoch, check_point_file):
    graph = tf.Graph()
    with graph.as_default():
        x = tf.placeholder("float", shape=[None, input_shape[0],input_shape[1],1], name="input")

        g_end, z, optimizer, cost,z_mean,z_log_sigma_sq = createGraph(x, False, True)

        path = os.path.join("outpy.avi")
        cap = cv2.VideoCapture(path)
        title = "Morphed/MorphPy_{}_{}.avi".format(n_z, epoch)
        out = cv2.VideoWriter(title, cv2.VideoWriter_fourcc('M','J','P','G'), 30, (64*2,64+64*2))

        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, check_point_file)
            #print("Model restored.")
            while(True):
                loop, data = cap.read()

                if(loop):
                    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
                    data = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)
                    
                    z_baseline,z_m,z_sigma  = sess.run((z, z_mean, z_log_sigma_sq), feed_dict={x:data.reshape(1,input_shape[0], input_shape[1], 1)/255})
                    resultImage = sess.run((g_end), feed_dict={z: z_baseline})
                    resultImage = resultImage*255
                    resultImage = resultImage.reshape((64,64))

                    
                    ## Create Gaussian Plot
                    fig = plt.figure()
                    for i in range(n_z):
                        mu = z_m[0][i]
                        variance = z_sigma[0][i]
                        #print(variance)
                        sigma = np.sqrt(variance)
                        #print(sigma.shape)
                        x_space = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
                        test = plt.plot(x_space, mlab.normpdf(x_space, mu, sigma))
                        #fig.add_subplot(111)
                        plt.plot(x_space, mlab.normpdf(x_space, mu, sigma))
                        #fig.annotate(i)
                    fig.canvas.draw()
                    gaussian_plot = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                    gaussian_plot = gaussian_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                    gaussian_plot = cv2.cvtColor(gaussian_plot, cv2.COLOR_RGB2GRAY)
                    gaussian_plot = cv2.resize(gaussian_plot, (128, 128)) 
                    plt.close()
                 
                    ## Create Canvas with input and output
                    canvas = np.zeros((input_shape[0]*3, input_shape[1]*2))
                    canvas[0:input_shape[1], 0:input_shape[1]] = data
                    canvas[0:input_shape[1],input_shape[1]:] = resultImage

                    #canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
                    
                    canvas[input_shape[1]:,0:] = gaussian_plot         
                    #gaussian_plot = cv2.cvtColor(gaussian_plot, cv2.COLOR_RGB2GRAY)
                    
                    
                    canvas = np.array(canvas, dtype=np.uint8)

                  
                    canvas = canvas.reshape((64*3,64*2,1))
                    grayscaled = np.zeros((canvas.shape[0], canvas.shape[1],3))
                    grayscaled[:, :,0] = canvas[:,:,0]
                    grayscaled[:, :,1] = canvas[:,:,0]
                    grayscaled[:, :,2] = canvas[:,:,0]
                    grayscaled = grayscaled.astype(np.uint8)
                    #plt.imshow(grayscaled,cmap=plt.get_cmap('gray'))
                    #plt.show()
                    
               
                    writeFrame = cv2.cvtColor(grayscaled, cv2.COLOR_RGB2BGR)
                    
            
                    ## Schreibe den Frame
                    out.write(writeFrame)
                   
                else:  
                    break
                    
    cap.release()
    out.release()
    cv2.destroyAllWindows()






def createNormalVideo(n_z, epoch, check_point_file):
    graph = tf.Graph()
    with graph.as_default():
        x = tf.placeholder("float", shape=[None, input_shape[0],input_shape[1],1], name="input")

        g_end, z, optimizer, cost,z_mean,z_log_sigma_sq = createGraph(x, False, True)

        path = os.path.join("outpy.avi")
        cap = cv2.VideoCapture(path)
        title = "Morphed/MorphPy_{}_{}.avi".format(n_z, epoch)
        out = cv2.VideoWriter(title, cv2.VideoWriter_fourcc('M','J','P','G'), 30, (64*2,64))

        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess, check_point_file)
            #print("Model restored.")
            while(True):
                loop, data = cap.read()

                if(loop):
                    data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
                    data = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)
                    
                    z_baseline,z_m,z_sigma  = sess.run((z, z_mean, z_log_sigma_sq), feed_dict={x:data.reshape(1,input_shape[0], input_shape[1], 1)/255})
                    resultImage = sess.run((g_end), feed_dict={z: z_baseline})
                    resultImage = resultImage*255
                    resultImage = resultImage.reshape((64,64))

                    
                 
                    ## Create Canvas with input and output
                    canvas = np.zeros((input_shape[0], input_shape[1]*2))
                    canvas[0:input_shape[1], 0:input_shape[1]] = data
                    canvas[0:input_shape[1],input_shape[1]:] = resultImage

                    #canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
                    
                    
                    canvas = np.array(canvas, dtype=np.uint8)

                  
                    canvas = canvas.reshape((64,64*2,1))
                    grayscaled = np.zeros((canvas.shape[0], canvas.shape[1],3))
                    grayscaled[:, :,0] = canvas[:,:,0]
                    grayscaled[:, :,1] = canvas[:,:,0]
                    grayscaled[:, :,2] = canvas[:,:,0]
                    grayscaled = grayscaled.astype(np.uint8)
                    #plt.imshow(grayscaled,cmap=plt.get_cmap('gray'))
                    #plt.show()
                    
               
                    writeFrame = cv2.cvtColor(grayscaled, cv2.COLOR_RGB2BGR)
                    
            
                    ## Schreibe den Frame
                    out.write(writeFrame)
                   
                else:  
                    break
                    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

