from __future__ import division
from __future__ import print_function
import os.path
import scipy.io
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib as plt
from tensorflow.examples.tutorials.mnist import input_data

#mnist = input_data.read_data_sets('MNIST')

# Data = np.array(scipy.io.loadmat('/home/eunho/Downloads/VAE-TensorFlow-master/Train_py101.mat')["ans"])
np.random.seed(0)
tf.set_random_seed(0)

data = scipy.io.loadmat('/home/eunsongk/Data/ADNI/ADNI_Sliced/Slice_1.mat')["Slice"]
# label = scipy.io.loadmat('/home/eunsongk/Data/Slice_1.mat')["Labels"]

arr = np.array(data[:, :])

#print (arr)
#print (arr.shape)
# # print (data.shape)
#Label = np.array(label[:, :])
# print (Label)

# Label = np.array(label[:, :])
# s = pd.Series(Label)
# onehot= pd.get_dummies(s)
# print (onehot)
# plt.imshow(arr)

n_samples = 62

input_dim = 116*1
hidden_encoder_dim = 200
hidden_decoder_dim = 200
latent_dim = 20
lam = 0


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.001)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0., shape=shape)
    return tf.Variable(initial)


x = tf.placeholder("float", shape=[None, input_dim])
#x = tf.placeholder("float", shape=[input_dim])
l2_loss = tf.constant(0.0)

W_encoder_input_hidden = weight_variable([input_dim, hidden_encoder_dim])
b_encoder_input_hidden = bias_variable([hidden_encoder_dim])
l2_loss += tf.nn.l2_loss(W_encoder_input_hidden)

# Hidden layer encoder
hidden_encoder = tf.nn.relu(tf.matmul(x, W_encoder_input_hidden) + b_encoder_input_hidden)

W_encoder_hidden_mu = weight_variable([hidden_encoder_dim, latent_dim])
b_encoder_hidden_mu = bias_variable([latent_dim])
l2_loss += tf.nn.l2_loss(W_encoder_hidden_mu)

# Mu encoder
mu_encoder = tf.matmul(hidden_encoder, W_encoder_hidden_mu) + b_encoder_hidden_mu


W_encoder_hidden_logvar = weight_variable([hidden_encoder_dim, latent_dim])
b_encoder_hidden_logvar = bias_variable([latent_dim])
l2_loss += tf.nn.l2_loss(W_encoder_hidden_logvar)

# Sigma encoder
logvar_encoder = tf.matmul(hidden_encoder, W_encoder_hidden_logvar) + b_encoder_hidden_logvar


# Sample epsilon
epsilon = tf.random_normal(tf.shape(logvar_encoder), name='epsilon')

# Sample latent variable
std_encoder = tf.exp(0.5 * logvar_encoder)
z = mu_encoder + tf.mul(std_encoder, epsilon)

W_decoder_z_hidden = weight_variable([latent_dim, hidden_decoder_dim])
b_decoder_z_hidden = bias_variable([hidden_decoder_dim])
l2_loss += tf.nn.l2_loss(W_decoder_z_hidden)

# Hidden layer decoder
hidden_decoder = tf.nn.relu(tf.matmul(z, W_decoder_z_hidden) + b_decoder_z_hidden)

W_decoder_hidden_reconstruction = weight_variable([hidden_decoder_dim, input_dim])
b_decoder_hidden_reconstruction = bias_variable([input_dim])
l2_loss += tf.nn.l2_loss(W_decoder_hidden_reconstruction)

KLD = -0.5 * tf.reduce_sum(1 + logvar_encoder - tf.pow(mu_encoder, 2) - tf.exp(logvar_encoder), reduction_indices=1)

x_hat = tf.matmul(hidden_decoder, W_decoder_hidden_reconstruction) + b_decoder_hidden_reconstruction
BCE = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(x_hat, x), reduction_indices=1)

loss = tf.reduce_mean(BCE + KLD)

regularized_loss = loss + lam * l2_loss

loss_summ = tf.summary.scalar("lowerbound", loss)
train_step = tf.train.AdamOptimizer(0.01).minimize(regularized_loss)

# add op for merging summary
summary_op = tf.summary.merge_all()
# add Saver ops
saver = tf.train.Saver()

n_steps = int(100)
batch_size = 10

######################################################################################################3
with tf.Session() as sess:
    # summary_writer = tf.summary.FileWriter('experiment', graph=sess.graph)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for step in xrange(1, n_steps):
        # mnist.train.next_batch(batch_size)
        #batch = mnist.train.next_batch(batch_size)

        indices = np.random.choice(n_samples, batch_size)
        #batch, _ = arr[indices]

        batch = arr[indices]
        # print (batch)
        print (batch.shape)
        # batch = np.reshape(batch, (batch_size, 19720))
        # batch = mnist[:, step*batch_size:(step+1)*batch_size]
        feed_dict = {x: batch[:]}
        _, cur_loss, summary_str = sess.run([train_step, loss, summary_op], feed_dict=feed_dict)
        # summary_writer.add_summary(summary_str, step)

        if step % 5 == 0:
         #   save_path = saver.save(sess, "save/model.ckpt")
            print("Step {0} | Loss: {1}".format(step, cur_loss))
sess.close()

