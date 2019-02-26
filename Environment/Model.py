from Config import *

import numpy as np
import tensorflow as tf
tfd = tf.contrib.distributions




# Erstelle den ComputationGraph für das Modell 
x = tf.placeholder("float", shape=[None, input_shape[0],input_shape[1],1], name="input")
def createGraph(x, random=True, full=False):
    # Schicht 1 mappt von ?x64x64 -> ?x32x32 mit 32 5x5 Filtern
    d_w1 = tf.get_variable('d_w1', [5, 5, 1, 32], initializer=tf.truncated_normal_initializer(stddev=0.02))
    d_b1 = tf.get_variable('d_b1', [32], initializer=tf.constant_initializer(0))
    d1 = tf.nn.conv2d(input=x, filter=d_w1, strides=[1, 1, 1, 1], padding='SAME')
    d1 = d1 + d_b1
    d1 = tf.nn.relu(d1)
    d1 = tf.nn.avg_pool(d1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Schicht 2 mappt von ?x32x32 -> ?x16x16 mit 64 5x5 Filtern
    # This finds 64 different 5 x 5 pixel features
    d_w2 = tf.get_variable('d_w2', [5, 5, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.02))
    d_b2 = tf.get_variable('d_b2', [64], initializer=tf.constant_initializer(0))
    d2 = tf.nn.conv2d(input=d1, filter=d_w2, strides=[1, 1, 1, 1], padding='SAME')
    d2 = d2 + d_b2
    d2 = tf.nn.relu(d2)
    d2 = tf.nn.avg_pool(d2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            
    #print(d2) # 16x16x64 = 16384

    # Schicht 3 mappt von ?x16x16 -> ?x8x8 mit 128 5x5 Filtern
    d_w3 = tf.get_variable('d_w3', [5,5,64,128], initializer=tf.truncated_normal_initializer(stddev=0.02))
    d_b3 = tf.get_variable('d_b3', [128], initializer=tf.constant_initializer(0))
    d3 = tf.nn.conv2d(input=d2, filter=d_w3, strides=[1, 1, 1, 1], padding='SAME')
    d3 = d3 + d_b3
    d3 = tf.nn.relu(d3)
    d3 = tf.nn.avg_pool(d3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    # Schicht 4 mappt von ?x8x8 = ?x8192 -> 512
    d_w4 = tf.get_variable('d_w4', [8*8*128, 512], initializer=tf.truncated_normal_initializer(stddev=0.02))
    d_b4 = tf.get_variable('d_b4', [512], initializer=tf.constant_initializer(0))
    # Arrangiere den Vektor neu
    d4 = tf.reshape(d3, [-1, 8*8*128])
    d4 = tf.matmul(d4, d_w4)
    d4 = d4 + d_b4
    d4 = tf.nn.relu(d4)

    # Schicht 5 mappt von 512 -> z_Dimension
    d_w5 = tf.get_variable('d_w5', [512, n_z], initializer=tf.truncated_normal_initializer(stddev=0.02))
    d_b5 = tf.get_variable('d_b5', [n_z], initializer=tf.constant_initializer(0))
    d5 = tf.add(tf.matmul(d4, d_w5), d_b5, name="d_result")
    d5 = tf.nn.relu(d5)



    # Die Vorhersagen für mu und sigma
    z_mean = tf.contrib.layers.fully_connected(d5, n_z, activation_fn=None)
    z_log_sigma_sq = tf.contrib.layers.fully_connected(d5, n_z, activation_fn=tf.nn.softplus)
    # Sample von der Dsitribution oder benutze nur mu um die Zufallskomponente zu entfernen (Das wird für die Videos benötigt)
    if random:
        eps = tf.random_normal((tf.shape(x)[0], n_z), 0, 1, dtype=tf.float32, name="eps")
    else:
        eps = tf.random_normal((tf.shape(x)[0], n_z), 0, 0, dtype=tf.float32, name="eps")
    # Reparametization-Trick
    z = tf.add(z_mean, tf.multiply(tf.sqrt(tf.exp(z_log_sigma_sq)), eps), name="z")  # The sampled z

    # Schicht 1 des Generators mappt von z_Dimension -> 16384
    g_w1 = tf.get_variable('g_w1', [n_z, 4*4*1024], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
    g_b1 = tf.get_variable('g_b1', [4*4*1024], initializer=tf.truncated_normal_initializer(stddev=0.02))
    g1 = tf.add(tf.matmul(z, g_w1), g_b1)
    # Arrangiere den Vektor neu in ein ?x4x4 mit 1024 Filter
    g1 = tf.reshape(g1, [-1, 4, 4, 1024])
    g1 = tf.layers.batch_normalization(g1, training=True)
    g1 = tf.nn.relu(g1)

    # Schicht 2 des Generators mappt von ?x4x4 -> ?x8x8 mit 512 Filter
    g2 = tf.layers.conv2d_transpose(g1, 512, [5, 5], strides=(2, 2), padding='SAME')
    g2 = tf.layers.batch_normalization(g2, training=True)
    g2 = tf.nn.relu(g2)

    # Schicht 3 des Generators mappt von ?x8x8 -> ?x16x16 mit 256 Filter
    g3 = tf.layers.conv2d_transpose(g2, 256, [5, 5], strides=(2, 2), padding='SAME')
    g3 = tf.layers.batch_normalization(g3, training=True)
    g3 = tf.nn.relu(g3)

    # Schicht 4 des Generators mappt von ?x16x16 -> ?x32x32 mit 128 Filter
    g4 = tf.layers.conv2d_transpose(g3, 128, [5, 5], strides=(2, 2), padding='SAME')
    g4 = tf.layers.batch_normalization(g4, training=True)
    g4 = tf.nn.relu(g4)

    # Schicht 5 des Generators mappt von ?x32x32 -> ?x64x64 mit 1 Filter
    g5 = tf.layers.conv2d_transpose(g4, 1, [5, 5], strides=(2, 2), padding='SAME')
    g5 = tf.nn.sigmoid(g5 , name="g_result")

    # Der Output des Generators
    g_end = g5
    

    # ELBO implementation
    elbo = -tf.square(x-g_end)
    elbo = tf.reduce_sum(elbo)

    
    # Wandle die Daten in eine statistische Verteilung um damit die KL-Abweichung berechnet werden kann
    helper = tf.distributions.Normal(loc=z_mean, scale=z_log_sigma_sq)
    prior = tf.distributions.Normal(loc=tf.zeros((tf.shape(x)[0], n_z)), scale=tf.ones((tf.shape(x)[0], n_z)))

    # Berechne die KL-Abweichung
    divergence = tfd.kl_divergence(helper, prior)
    divergence = tf.reduce_sum(divergence,1)

    # Setze die beiden Terme zusammem
    cost = tf.reduce_mean(elbo - divergence)
    cost = -cost
    
    # Der Optimierer für das Gradientenabstiegsverfahren
    optimizer =  tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(cost)

    # Liefere die Keyelements von den Graphen zurück
    if full:
        return g_end, z, optimizer, cost, z_mean, z_log_sigma_sq
    return g_end, z, optimizer, cost



