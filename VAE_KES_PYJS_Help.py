from __future__ import division
from __future__ import print_function
import os.path
import scipy.io
from sklearn.preprocessing import normalize
import numpy as np
from numpy import linalg
from numpy.linalg import norm
import tensorflow as tf
import pandas as pd
import matplotlib as plt

import random

np.random.seed(0)
tf.set_random_seed(0)
data = scipy.io.loadmat('/home/eunsongk/Data/ADNI/ADNI_eMCI_AAL116ROI_TimeSeries(0.01-0.08).mat')["aal116ROITimeSeries"]
labels = scipy.io.loadmat('/home/eunsongk/Data/ADNI/ADNI_eMCI_AAL116ROI_TimeSeries(0.01-0.08).mat')["labels"]
# print(data.shape)
#(62,1)
labels = np.array([labels]*130).transpose()
labels = np.asarray(labels).ravel()
labels = np.asarray(labels)


#(8060,)


arr = np.empty((data.shape[0], data[0, 0].shape[0], data[0, 0].shape[1]))
# print(arr.shape)
#(62, 130, 116)

for dat, cnt in zip(data[:, 0], np.arange(arr.shape[0])):
    arr[cnt] = dat
arr = arr.reshape(-1, arr.shape[-1])
# print(arr.shape)
# (8060,116)

#Gaussian normalization
arr-=np.mean(arr)
arr/=np.std(arr)

# norm1 = x / np.linalg.norm(x)
# norm2 = normalize(arr[:, np.newaxis], axis=1).ravel()
# print(norm2)
# print np.all(norm1 == norm2)

#norm1 = arr / np.linalg.norm(arr)
norm2 = normalize(arr, norm='l2', axis=1, copy=True, return_norm=False)
arr = norm2

# random.shuffle(arr)
# train_data = arr[:6448]
# test_data = arr[6448:]
#
# arr = train_data
# arr2 = test_data
#
# # random.shuffle(labels)
# training_label = labels[:6448]
# test_label = labels[6448:]
# test_label[test_label==-1]=0

# test_label = labels
# test_label[test_label==-1]=0
#
#
# # Label one hot encoding
# def one_hot(j):
#     e = np.zeros(2)
#     e[j] = 1
#     return e
# test_label = [one_hot(int(x)) for x in test_label[:]]
#
# # test_label = np.asarray(test_label)
# labels = np.array(test_label)
#
#
# random.shuffle(arr)
# train_data = arr[:6448]
# test_data = arr[6448:]
#
# arr = train_data
# arr2 = test_data
#
# random.shuffle(labels)
# training_label = labels[:6448]
# test_label = labels[6448:]
#
# input_dim = 116
# n_samples = arr.shape[0]
# # # 6448
# n_samples2 = arr2.shape[0]
# # # 1612

###################################################################################################################
# label = scipy.io.loadmat('/home/eunsongk/Data/Slice_1.mat')["Labels"]
# arr = np.empty((data.shape[0], data[0, 0].shape[0], data[0, 0].shape[1]))
# for dat, cnt in zip(data[:, 0], np.arange(arr.shape[0])):
#     arr[cnt] = dat
# print(arr.shape)
# arr = arr.reshape(-1, arr.shape[-1])
# print(arr.shape)
#
# arr-=np.mean(arr)
# arr/=np.std(arr)

#print (arr)
# #print (arr.shape)
# # # print (data.shape)
# #Label = np.array(label[:, :])
# # print (Label)
#######################################################################################################################


n_samples = 8060
input_dim = 116
hidden_encoder_dim = 50
hidden_decoder_dim = 50
latent_dim = 20
lam = 0
batch_size = 4


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.001)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0., shape=shape)
    return tf.Variable(initial)


#x = tf.placeholder("float", shape=[batch_size, input_dim])
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
train_step = tf.train.AdamOptimizer(0.0001).minimize(regularized_loss)

# add op for merging summary
summary_op = tf.summary.merge_all()
# add Saver ops
saver = tf.train.Saver()

n_steps = int(100)

######################################################################################################3
with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter('/tmp/log', graph=sess.graph)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # for step in xrange(1, n_steps):
    #     # mnist.train.next_batch(batch_size)
    #     #batch = mnist.train.next_batch(batch_size)
    #
    #     indices = np.random.choice(n_samples, batch_size)
    #     #batch, _ = arr[indices]
    #
    #     batch = arr[indices]
    #     # print (batch)
    #     print (batch.shape)
    #     # batch = np.reshape(batch, (batch_size, 19720))
    #     # batch = mnist[:, step*batch_size:(step+1)*batch_size]




#JS
  #   for step in range(n_steps*(n_samples//batch_size)):
  #       offset = (step)%(n_samples//batch_size)
  #       batch = arr[offset*batch_size: (offset+1)*batch_size]
  #       feed_dict = {x: batch}
  #       _, cur_loss, summary_str = sess.run([train_step, loss, summary_op], feed_dict=feed_dict)
  #
  #       # summary_writer.add_summary(summary_str, step)
  #
  #       if step % 1 == 0:
  # # save_path = saver.save(sess, "save/model.ckpt")
  #
  #           print("Step {0} | Loss: {1}".format(step, cur_loss))



    for step in range(n_steps*(n_samples//batch_size)):
        offset = (step)%(n_samples//batch_size)
        batch = arr[offset*batch_size: (offset+1)*batch_size]
        feed_dict = {x: batch}
        _, cur_loss, summary_str = sess.run([train_step, loss, summary_op], feed_dict=feed_dict)

        # summary_writer.add_summary(summary_str, step)

        if step % 100 == 0:
  # save_path = saver.save(sess, "save/model.ckpt")

            print("Step {0} | Loss: {1}".format(step, cur_loss))


sess.close()

