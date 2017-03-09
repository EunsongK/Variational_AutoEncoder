import itertools
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from matplotlib.path import Path
import matplotlib.patches as patches
import pandas as pd

from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

from medpy.io import load

import numpy as np

import os
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.distributions as distributions



import scipy.io
from scipy.misc import imsave
from scipy.sparse import coo_matrix

# import random
import time
import tsne

# For MCI dataset
# Original file was mat file


data = scipy.io.loadmat('/home/eunsongk/Data/ADNI/ADNI_eMCI_AAL116ROI_TimeSeries(0.01-0.08).mat')["aal116ROITimeSeries"]
labels = scipy.io.loadmat('/home/eunsongk/Data/ADNI/ADNI_eMCI_AAL116ROI_TimeSeries(0.01-0.08).mat')["labels"]
# print(data.shape)
# (62,1)
labels = np.array([labels] * 130).transpose()
labels = np.asarray(labels).ravel()
labels = np.asarray(labels)

# (8060,)


arr = np.empty((data.shape[0], data[0, 0].shape[0], data[0, 0].shape[1]))
# print(arr.shape)
# (62, 130, 116)

for dat, cnt in zip(data[:, 0], np.arange(arr.shape[0])):
    arr[cnt] = dat
arr = arr.reshape(-1, arr.shape[-1])
# print(arr.shape)
# (8060,116)

# Gaussian normalization
arr -= np.mean(arr)
arr /= np.std(arr)

# norm1 = x / np.linalg.norm(x)
# norm2 = normalize(arr[:, np.newaxis], axis=1).ravel()
# print(norm2)
# print np.all(norm1 == norm2)

# norm1 = arr / np.linalg.norm(arr)
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

test_label_tSNE = labels
test_label_tSNE[test_label_tSNE == -1] = 0


# Label one hot encoding
def one_hot(j):
    e = np.zeros(2)
    e[j] = 1
    return e


labels = [one_hot(int(x)) for x in test_label_tSNE[:]]

test_label = np.asarray(labels)
labels = np.array(test_label)


input_dim = 116
n_samples = arr.shape[0]




# For plot
# sns.set_style('whitegrid')
# sns.set_context("notebook", font_scale=1.5,
#                 rc={"lines.linewidth": 2.5})

sg = tf.contrib.bayesflow.stochastic_graph
st = tf.contrib.bayesflow.stochastic_tensor


flags = tf.app.flags
flags.DEFINE_string('data_dir', '/tmp/dat/', 'Directory for data')
flags.DEFINE_string('logdir', '/tmp/log/', 'Directory for logs')

# For making plots:
# flags.DEFINE_integer('latent_dim', 2, 'Latent dimensionality of model')
# flags.DEFINE_integer('batch_size', 64, 'Minibatch size')
# flags.DEFINE_integer('n_samples', 10, 'Number of samples to save')
# flags.DEFINE_integer('print_every', 10, 'Print every n iterations')
# flags.DEFINE_integer('hidden_size', 200, 'Hidden size for neural networks')
# flags.DEFINE_integer('n_iterations', 1000, 'number of iterations')

# For bigger model:
flags.DEFINE_integer('latent_dim', 10, 'Latent dimensionality of model')
flags.DEFINE_integer('batch_size', 130, 'Minibatch size')
flags.DEFINE_integer('n_samples', 100, 'Number of samples to save')
flags.DEFINE_integer('print_every', 20, 'Print every n iterations')
flags.DEFINE_integer('hidden_size', 30, 'Hidden size for neural networks')
flags.DEFINE_integer('n_iterations', 1000, 'number of iterations')

FLAGS = flags.FLAGS


def inference_network(x, latent_dim, hidden_size):
    """Construct an inference network parametrizing a Gaussian.

    Args:
      x: A batch of MNIST digits.
      latent_dim: The latent dimensionality.
      hidden_size: The size of the neural net hidden layers.

    Returns:
      mu: Mean parameters for the variational family Normal
      sigma: Standard deviation parameters for the variational family Normal
    """
    with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
        net = slim.flatten(x)
        net = slim.fully_connected(net, hidden_size)
       ###########Add one more hidden layer##########################
        # net = slim.fully_connected(net, hidden_size)
        ###########Add one more hidden layer##########################
        net = slim.fully_connected(net, hidden_size)
        gaussian_params = slim.fully_connected(net, latent_dim * 2, activation_fn=None)

        # The mean parameter is unconstrained
        mu = gaussian_params[:, :latent_dim]
        # The standard deviation must be positive. Parametrize with a softplus and
        # add a small epsilon for numerical stability
        sigma = 1e-6 + tf.nn.softplus(gaussian_params[:, latent_dim:])
        return mu, sigma


def generative_network(z, hidden_size):
    """Build a generative network parametrizing the likelihood of the data

    Args:
      z: Samples of latent variables
      hidden_size: Size of the hidden state of the neural net

    Returns:
      mu: Mean parameters for the variational family Normal
      sigma: Standard deviation parameters for the variational family Normal
    """

    with slim.arg_scope([slim.fully_connected], activation_fn=tf.nn.relu):
        net = slim.fully_connected(z, hidden_size)
        ###########Add one more hidden layer##########################
        # net = slim.fully_connected(net, hidden_size)
        ###########Add one more hidden layer##########################
        net = slim.fully_connected(net, hidden_size)
        gaussian_params = slim.fully_connected(net, input_dim * 2, activation_fn=None)
        mu = gaussian_params[:, :input_dim]
        sigma = 1e-6 + tf.nn.softplus(gaussian_params[:, input_dim:])
        # mu = tf.reshape(mu, [-1, 116, 1])
        mu = tf.reshape(mu, [-1, 116])
        # sigma = tf.reshape(sigma, [-1, 116, 1])
        sigma = tf.reshape(sigma, [-1, 116])
        return mu, sigma


def TimeTrajectory_foroneZ(a, b):
    stacked_q_mu = None
    stacked_q_sigma = None
    for s in xrange(FLAGS.batch_size):
        Sparse_q_mu = tf.SparseTensor(
            indices=[[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9]],
            values=a[s], shape=[FLAGS.latent_dim, FLAGS.latent_dim])
        Sparse_q_sigma = tf.SparseTensor(
            indices=[[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6], [7, 7], [8, 8], [9, 9]],
            values=b[s], shape=[FLAGS.latent_dim, FLAGS.latent_dim])

        Densed_q_mu = tf.sparse_tensor_to_dense(Sparse_q_mu, default_value=0, validate_indices=True, name=None)
        Densed_q_sigma = tf.sparse_tensor_to_dense(Sparse_q_sigma, default_value=0, validate_indices=True, name=None)
        if s==0:
            stacked_q_mu = Densed_q_mu
            stacked_q_sigma = Densed_q_sigma
        else:
            stacked_q_mu = tf.concat_v2(values=[stacked_q_mu, Densed_q_mu], axis=0)
            stacked_q_sigma = tf.concat_v2(values=[stacked_q_sigma, Densed_q_sigma], axis=0)
        # print(stacked_q_sigma.get_shape())
    return stacked_q_mu, stacked_q_sigma


def train():
    # Input placeholders
    with tf.name_scope('arr'):
        x = tf.placeholder(tf.float32, [FLAGS.batch_size, input_dim])


    with tf.variable_scope('variational'):
        q_mu, q_sigma = inference_network(x=x,
                                          latent_dim=FLAGS.latent_dim,
                                          hidden_size=FLAGS.hidden_size)

        Onez_q_mu, Onez_q_sigma = TimeTrajectory_foroneZ(a=q_mu, b=q_sigma)

        Reshaped_q_mu = tf.reshape(Onez_q_mu, [FLAGS.batch_size, FLAGS.latent_dim])
        Reshaped_q_sigma = tf.reshape(Onez_q_sigma, [FLAGS.batch_size, FLAGS.latent_dim])



        with st.value_type(st.SampleValue()):
            # The variational distribution is a Normal with mean and standard
            # deviation given by the inference network
            q_z = st.StochasticTensor(distributions.MultivariateNormalDiag(mu=Reshaped_q_mu, diag_stdev=Reshaped_q_sigma))



    with tf.variable_scope('model'):
        # The likelihood is Bernoulli-distributed with logits given by the generative network
        p_x_given_z_mu, p_x_given_z_sigma = generative_network(z=q_z,
                                                               hidden_size=FLAGS.hidden_size)
        p_x_given_z = distributions.MultivariateNormalDiag(mu=p_x_given_z_mu, diag_stdev=p_x_given_z_sigma)


    # Take samples from the prior
    with tf.variable_scope('model', reuse=True):
        p_z = distributions.MultivariateNormalDiag(mu=np.zeros(FLAGS.latent_dim, dtype=np.float32),
                                                   diag_stdev=np.ones(FLAGS.latent_dim, dtype=np.float32))
        p_z_sample = p_z.sample_n(FLAGS.n_samples)
        p_x_given_z_mu, p_x_given_z_sigma = generative_network(z=p_z_sample,
                                                               hidden_size=FLAGS.hidden_size)
        prior_predictive = distributions.MultivariateNormalDiag(mu=p_x_given_z_mu, diag_stdev=p_x_given_z_sigma)

        # Take samples from the prior with a placeholder
        # with tf.variable_scope('model', reuse=True):
        #   z_input = tf.placeholder(tf.float32, [None, FLAGS.latent_dim])
        #   p_x_given_z_mu, p_x_given_z_sigma= generative_network(z=z_input,
        #                                           hidden_size=FLAGS.hidden_size)
        #   prior_predictive_inp = distributions.MultivariateNormalDiag(mu=p_x_given_z_mu, diag_stdev = p_x_given_z_sigma)
        #
        #   prior_predictive_inp_sample = prior_predictive_inp.sample()

        #################################################################################################

        # Build the evidence lower bound (ELBO) or the negative loss
        kl = distributions.kl(q_z.distribution, p_z)
        expected_log_likelihood = tf.reduce_sum(p_x_given_z.log_prob(x), -1)
        elbo = tf.reduce_sum(expected_log_likelihood - kl, 0)

        optimizer = tf.train.RMSPropOptimizer(learning_rate=0.00001)
        train_op = optimizer.minimize(-elbo)
        # train_op = optimizer.minimize(elbo)
        tf.scalar_summary("ELBO", elbo)


        # Merge all the summaries
        summary_op = tf.summary.merge_all()
        init_op = tf.global_variables_initializer()

        # Run training
        sess = tf.InteractiveSession()
        sess.run(init_op)

        print('Saving TensorBoard summaries and images to: %s' % FLAGS.logdir)
        train_writer = tf.summary.FileWriter(FLAGS.logdir, sess.graph)



        for i in range(FLAGS.n_iterations * (n_samples // FLAGS.batch_size)):
            offset = (i) % (n_samples // FLAGS.batch_size)
            # Re-binarize the data at every batch; this improves results
            # Original
            # np_x_fixed = arr2[offset * FLAGS.batch_size:(offset + 1) * FLAGS.batch_size].reshape(-1, input_dim)
            np_x = arr[offset * FLAGS.batch_size:(offset + 1) * FLAGS.batch_size].reshape(-1, input_dim)
            np_y_fixed = labels[offset * FLAGS.batch_size:(offset + 1) * FLAGS.batch_size]

            # _, q_mu_out, q_sigma_out = sess.run([train_op, q_mu, q_sigma], {x: np_x})
            sess.run(train_op, {x: np_x})


            t0 = time.time()
            if i % FLAGS.print_every == 0:
                np_elbo, summary_str = sess.run([elbo, summary_op], {x: np_x})
                train_writer.add_summary(summary_str, i)
                print('Iteration: {0:d} ELBO: {1:.3f} Examples/s: {2:.3e}'.format(i, np_elbo / FLAGS.batch_size,

                                                                                  FLAGS.batch_size * FLAGS.print_every / (
                                                                                      time.time() - t0)))

                t0 = time.time()


            if i in range((FLAGS.n_iterations-1) *(n_samples//FLAGS.batch_size), (FLAGS.n_iterations) *(n_samples//FLAGS.batch_size)):

                sess.run([q_mu, q_sigma)

#             # print(range((FLAGS.n_iterations-1) *(n_samples//FLAGS.batch_size), (FLAGS.n_iterations) *(n_samples//FLAGS.batch_size)))
#             if i in range((FLAGS.n_iterations-1) *(n_samples//FLAGS.batch_size), (FLAGS.n_iterations) *(n_samples//FLAGS.batch_size)):
#               # if offset==0 and i!=0:
#
#                 print "Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset."
#                 print "Running example of_input ADNI..."
#                 # X = sess.run(q_mu, {x: np_x})
#                 X, q_sigma_out = sess.run([q_mu, q_sigma], {x: np_x})
#
#                 # np.savetxt('inferenced_z_mu_%d'%i, X)
#                 # np.savetxt('inferenced_z_sigma_%d'%i, q_sigma_out)
#
#                 labels_tsne = np.argmax(np_y_fixed, 1)
#
#                 Y = tsne.tsne(X, 2, 20, 15.0)
#                 # np.save('Error.npy', Y)
#                 # cmap = plt.get_cmap('bwr')
#
#                 fig = plt.figure(facecolor="white", figsize=(8.0, 6.0))
#                 plt.xlim(-50.0, 50.0)
#                 plt.ylim(-50.0, 50.0)
#                 plt.axis("off")
#
#                 if labels_tsne[0] == 1:
#                   plt.scatter(Y[:, 0], Y[:, 1], 20, c = labels_tsne, cmap=mpl.colors.ListedColormap('red'))
#                 else:
#                   plt.scatter(Y[:, 0], Y[:, 1], 20, c= labels_tsne, cmap=mpl.colors.ListedColormap('blue'))
#
#
#                 plt.savefig(os.path.join(FLAGS.logdir, 'tSNE_map_frame_%d.png' % i))
#
#
# ##########################################################################################################################################################
#                 # fig = plt.figure(facecolor="white", figsize=(15.0, 10.0))
#                 # scat = plt.scatter(Y[:, 0], Y[:, 1], 20, c = labels_tsne, cmap=mpl.colors.ListedColormap('black'))
#                 scat = plt.scatter([], [], c='white')
#
#                 def initiation():
#                     scat.set_offsets([])
#                     return scat,
#
#                 def animate(t):
#                     x_ani = Y[:, 0].transpose()
#                     y_ani = Y[:, 1].transpose()
#                     data_ani = np.hstack((x_ani[t:, np.newaxis], y_ani[t:, np.newaxis]))
#                     # print (data_ani)
#                     scat.set_offsets(data_ani)
#                     return scat,
#
# #                 # ims = []
# #                 # timepoint = []
# #                 #
# #                 # for a in scat():
# #                 #     timepoint.append(a)
# #                 #     ims.append(timepoint)
# #
#                 ani = animation.FuncAnimation(fig, animate, init_func=initiation, frames=FLAGS.batch_size + 17 , interval=200, blit=True)
#                 # plt.show()
#
#                 Writer = animation.writers['ffmpeg']
#                 writer = Writer(fps=13, metadata=dict(artist='Kang, Eun Song (Korea University MiLab)'), bitrate=1800)
#
#                 # ani.save("test_%d.mov" %i, writer=writer, dpi=300)
#                 ani.save(os.path.join(FLAGS.logdir, 'test_%d.mov'%i), writer=writer, dpi=300)
# ##############################################################################################################################################################
# #
# #                 #
# #                 # labels_plt = ['{0}'.format(j) for j in range(129)]
# #                 # for label, a, b in zip(labels_plt, Y[:, 0], Y[:, 1]):
# #                 #     plt.annotate(
# #                 #         label, xy=(a, b), xytext=(-0.07, 0.07),
# #                 #         textcoords='offset points', ha='right', va='bottom',
# #                 #         arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
# #                 #
# #                 plt.savefig(os.path.join(FLAGS.logdir, 'tSNE_map_frame_%d.png' % i))
#                 plt.close()
# #




def main(_):
    if tf.gfile.Exists(FLAGS.logdir):
        tf.gfile.DeleteRecursively(FLAGS.logdir)
    tf.gfile.MakeDirs(FLAGS.logdir)
    train()


if __name__ == '__main__':
    tf.app.run()