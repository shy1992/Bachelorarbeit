from Config import *

import numpy as np
import tensorflow as tf
tfd = tf.contrib.distributions




x = tf.placeholder("float", shape=[None, input_shape[0],input_shape[1],1], name="input")
def createGraph(x, random=True, full=False):
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

    if random:
        eps = tf.random_normal((tf.shape(x)[0], n_z), 0, 1, dtype=tf.float32, name="eps") # Adding a random number
    else:
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
    #print(g1)


    ## DeconvLayer #1 
    g2 = tf.layers.conv2d_transpose(g1, 512, [5, 5], strides=(2, 2), padding='SAME')
    g2 = tf.layers.batch_normalization(g2, training=True)
    g2 = tf.nn.relu(g2)
    #print(g2)


    g3 = tf.layers.conv2d_transpose(g2, 256, [5, 5], strides=(2, 2), padding='SAME')
    g3 = tf.layers.batch_normalization(g3, training=True)
    g3 = tf.nn.relu(g3)
    #print(g3)


    g4 = tf.layers.conv2d_transpose(g3, 128, [5, 5], strides=(2, 2), padding='SAME')
    g4 = tf.layers.batch_normalization(g4, training=True)
    g4 = tf.nn.relu(g4)
    #print(g4)

    g5 = tf.layers.conv2d_transpose(g4, 1, [5, 5], strides=(2, 2), padding='SAME')
    #g5 = tf.layers.batch_normalization(g5, training=True)
    g5 = tf.nn.sigmoid(g5 , name="g_result")
    #print(g5)

    g_end = g5
    

    #g_shot = tf.reshape(g_end, (-1, 64*64))
    #likelihood = tfd.Independent(tfd.Bernoulli(g_end), 3)
    #likelihood = tfd.Bernoulli(g_end)
    #likelihood = likelihood.log_prob(x)
    
    #x_in = tfd.Bernoulli(x)
    #x_out = tfd.Bernoulli(g_end)
    #likelihood = tfd.kl_divergence(x_in, x_out)
    
    #likelihood = -tf.abs(x-g_end)
    #likelihood = tf.reduce_sum(likelihood,1)

    likelihood = -tf.square(x-g_end)
    #print(likelihood)
    #likelihood = tf.reduce_sum(likelihood)
    likelihood = tf.reduce_mean(likelihood)
    #print(likelihood)
    
    #helper = tf.random_normal((tf.shape(x)[0], n_z), z_mean, z_log_sigma_sq, dtype=tf.float32)
    helper = tf.distributions.Normal(loc=z_mean, scale=z_log_sigma_sq)
    prior = tf.distributions.Normal(loc=tf.zeros((tf.shape(x)[0], n_z)), scale=tf.ones((tf.shape(x)[0], n_z)))

    #print("Helper", helper)
    #print("prior", prior)
    divergence = tfd.kl_divergence(helper, prior)
    divergence = tf.reduce_sum(divergence,1)

    #print(divergence)
    #print(likelihood.shape)
    elbo = tf.reduce_mean(likelihood - divergence)

    cost = -elbo
    
    # Use ADAM optimizer
    #optimizer =  tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
    optimizer =  tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)
    if full:
        return g_end, z, optimizer, cost, z_mean, z_log_sigma_sq
    return g_end, z, optimizer, cost



"""

with tf.Session() as sess_test:
    init = tf.global_variables_initializer()
    sess_test.run(init)
    print(x_dataset.shape)
    t, t2 = sess_test.run([divergence, z], feed_dict={z: np.zeros((1,n_z)), x: x_dataset[2:3].reshape(-1,input_shape[0],input_shape[1],1)/255})
    print("Likelihood shape",t.shape)
    ## Test runs
    t, t2 = sess_test.run([g_end, z], feed_dict={z: np.zeros((1,n_z)), x: x_dataset[2:3].reshape(-1,input_shape[0],input_shape[1],1)/255})

    t, t2 = sess_test.run([g_end, z], feed_dict={z: np.zeros((2,n_z))})
    t, t2 = sess_test.run([divergence, z], feed_dict={z: np.zeros((4,n_z)), x: x_dataset[2:5].reshape(-1,input_shape[0],input_shape[1],1)/255})
    print("Likelihood shape",t.shape)
    t = sess_test.run(g_end, feed_dict={x: x_dataset[2:3].reshape(-1,input_shape[0],input_shape[1],1)/255})
    print(t.shape)
    
plt.imshow(t.reshape(input_shape[0],input_shape[1])*255,cmap=plt.get_cmap('gray'))
plt.show()

"""

if __main__ == "__main__":
    g_end, z, optimizer, cost = createGraph(x)
    with tf.Session() as sess_test:
        init = tf.global_variables_initializer()
        sess_test.run(init)
        
        print(x_dataset.shape)
        t, t2 = sess_test.run([divergence, z], feed_dict={z: np.zeros((1,n_z)), x: x_dataset[2:3].reshape(-1,input_shape[0],input_shape[1],1)/255})
        print("Likelihood shape",t.shape)
        ## Test runs
        t, t2 = sess_test.run([g_end, z], feed_dict={z: np.zeros((1,n_z)), x: x_dataset[2:3].reshape(-1,input_shape[0],input_shape[1],1)/255})

        t, t2 = sess_test.run([g_end, z], feed_dict={z: np.zeros((2,n_z))})
        t, t2 = sess_test.run([divergence, z], feed_dict={z: np.zeros((4,n_z)), x: x_dataset[2:5].reshape(-1,input_shape[0],input_shape[1],1)/255})
        print("Likelihood shape",t.shape)
        t = sess_test.run(g_end, feed_dict={x: x_dataset[2:3].reshape(-1,input_shape[0],input_shape[1],1)/255})
        print(t.shape)
    
    #plt.imshow(t.reshape(input_shape[0],input_shape[1])*255,cmap=plt.get_cmap('gray'))
    #plt.show()
