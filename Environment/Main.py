
import tensorflow as tf
import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

#import PIL
import h5py
import math
from random import randint, shuffle
import time


tf.reset_default_graph()
input_shape = (64,64)

n_z = 30




x = tf.placeholder("float", shape=[None, input_shape[0],input_shape[1],1], name="input")

d_w1 = tf.get_variable('d_w1', [5, 5, 1, 32], initializer=tf.truncated_normal_initializer(stddev=0.02))
d_b1 = tf.get_variable('d_b1', [32], initializer=tf.constant_initializer(0))
d1 = tf.nn.conv2d(input=x, filter=d_w1, strides=[1, 1, 1, 1], padding='SAME')
d1 = d1 + d_b1
d1 = tf.nn.relu(d1)
d1 = tf.nn.avg_pool(d1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Second convolutional and pool layers
# This finds 64 different 5 x 5 pixel features
d_w2 = tf.get_variable('d_w2', [5, 5, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.02))
d_b2 = tf.get_variable('d_b2', [64], initializer=tf.constant_initializer(0))
d2 = tf.nn.conv2d(input=d1, filter=d_w2, strides=[1, 1, 1, 1], padding='SAME')
d2 = d2 + d_b2
d2 = tf.nn.relu(d2)
d2 = tf.nn.avg_pool(d2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
#print(d2) # 16x16x64 = 16384

# First fully connected layer
d_w3 = tf.get_variable('d_w3', [5,5,64,128], initializer=tf.truncated_normal_initializer(stddev=0.02))
d_b3 = tf.get_variable('d_b3', [128], initializer=tf.constant_initializer(0))
d3 = tf.nn.conv2d(input=d2, filter=d_w3, strides=[1, 1, 1, 1], padding='SAME')
d3 = d3 + d_b3
d3 = tf.nn.relu(d3)
d3 = tf.nn.avg_pool(d3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

d_w4 = tf.get_variable('d_w4', [8*8*128, 512], initializer=tf.truncated_normal_initializer(stddev=0.02))
d_b4 = tf.get_variable('d_b4', [512], initializer=tf.constant_initializer(0))
d4 = tf.reshape(d3, [-1, 8*8*128])
d4 = tf.matmul(d4, d_w4)
d4 = d4 + d_b4
d4 = tf.nn.relu(d4)

# Second fully connected layer
d_w5 = tf.get_variable('d_w5', [512, n_z], initializer=tf.truncated_normal_initializer(stddev=0.02))
d_b5 = tf.get_variable('d_b5', [n_z], initializer=tf.constant_initializer(0))
d5 = tf.add(tf.matmul(d4, d_w5), d_b5, name="d_result")
d5 = tf.nn.relu(d5)




z_mean = tf.contrib.layers.fully_connected(d5, n_z, activation_fn=None)
z_log_sigma_sq = tf.contrib.layers.fully_connected(d5, n_z, activation_fn=tf.nn.softplus)


eps = tf.random_normal((tf.shape(x)[0], n_z), 0, 0, dtype=tf.float32, name="eps") # Adding a random number
z = tf.add(z_mean, tf.multiply(tf.sqrt(tf.exp(z_log_sigma_sq)), eps), name="z")  # The sampled z





alpha = 0.2

g_w1 = tf.get_variable('g_w1', [n_z, 4*4*1024], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
g_b1 = tf.get_variable('g_b1', [4*4*1024], initializer=tf.truncated_normal_initializer(stddev=0.02))
g1 = tf.add(tf.matmul(z, g_w1), g_b1)
g1 = tf.reshape(g1, [-1, 4, 4, 1024])
g1 = tf.layers.batch_normalization(g1, training=True)
#g1 = tf.contrib.layers.batch_norm(g1, epsilon=1e-5, scope='g_b1')
g1 = tf.nn.relu(g1)
print(g1)


## DeconvLayer #1 
g2 = tf.layers.conv2d_transpose(g1, 512, [5, 5], strides=(2, 2), padding='SAME')
g2 = tf.layers.batch_normalization(g2, training=True)
g2 = tf.nn.relu(g2)
print(g2)


g3 = tf.layers.conv2d_transpose(g2, 256, [5, 5], strides=(2, 2), padding='SAME')
g3 = tf.layers.batch_normalization(g3, training=True)
g3 = tf.nn.relu(g3)
print(g3)


g4 = tf.layers.conv2d_transpose(g3, 128, [5, 5], strides=(2, 2), padding='SAME')
g4 = tf.layers.batch_normalization(g4, training=True)
g4 = tf.nn.relu(g4)
print(g4)

g5 = tf.layers.conv2d_transpose(g4, 1, [5, 5], strides=(2, 2), padding='SAME')
#g5 = tf.layers.batch_normalization(g5, training=True)
g5 = tf.nn.sigmoid(g5 , name="g_result")
print(g5)

g_end = g5


# Cost function
#x_reconstr_mean_ = tf.clip_by_value(g_end, 1e-7, 1-1e-7)
"""
x_reconstr_mean_ = g_end
print(x_reconstr_mean_)
print(x)
reconstr_loss = -tf.reduce_sum(x * tf.log(1e-10 + x_reconstr_mean_) + (1-x) * tf.log(1e-10 + 1 - x_reconstr_mean_), 1)
latent_loss = -0.5 * tf.reduce_sum(1 + z_log_sigma_sq - tf.square(z_mean) - tf.exp(z_log_sigma_sq), 1)
cost = tf.reduce_mean(reconstr_loss + latent_loss)   # average over batch
"""
tfd = tf.contrib.distributions

print(g_end)
#g_shot = tf.reshape(g_end, (-1, 64*64))
likelihood = tfd.Independent(tfd.Bernoulli(g_end), 3)
likelihood = likelihood.log_prob(x)
#helper = tf.random_normal((tf.shape(x)[0], n_z), z_mean, z_log_sigma_sq, dtype=tf.float32)
helper = tf.distributions.Normal(loc=z_mean, scale=z_log_sigma_sq)
prior = tf.distributions.Normal(loc=tf.zeros((tf.shape(x)[0], n_z)), scale=tf.ones((tf.shape(x)[0], n_z)))
print("Helper", helper)
print("prior", prior)
divergence = tfd.kl_divergence(helper, prior)
divergence = tf.reduce_sum(divergence,1)

print(divergence)
print(likelihood.shape)
elbo = tf.reduce_mean(likelihood - divergence)

cost = -elbo

# Use ADAM optimizer
optimizer =  tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)





























check_point_file = "model/modelHandConv.ckpt"

path = os.path.join("outpy.avi")
cap = cv2.VideoCapture(path)

out = cv2.VideoWriter('morphpyWithGaussian.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (64*3,64))



"""
loop, data = cap.read()
data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
data = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)
x_dataset = np.array([data], dtype=np.float32)
"""
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, check_point_file)
    print("Model restored.")
    while(True):
        loop, data = cap.read()

        if(loop):
            data = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
            data = cv2.cvtColor(data, cv2.COLOR_RGB2GRAY)
            
            z_baseline, z_m,  z_sigma = sess.run((z, z_mean, z_log_sigma_sq), feed_dict={x:data.reshape(1,input_shape[0], input_shape[1], 1)/255})
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
            gaussian_plot = cv2.resize(gaussian_plot, (64, 64)) 
            plt.close()
            
            #print(gaussian_plot.shape)
         
            ## Create Canvas with input and output
            canvas = np.empty((input_shape[0], input_shape[1]*3))
            canvas[0:input_shape[1], 0:input_shape[1]] = data
            canvas[0:input_shape[1],input_shape[1]:input_shape[1]*2] = resultImage
            canvas[0:input_shape[1],input_shape[1]*2:] = gaussian_plot

            
                
            canvas = np.array(canvas, dtype=np.uint8)
    
            canvas = canvas.reshape((64,64*3,1))
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

