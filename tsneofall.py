import os
import numpy as np
import tsne
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
from matplotlib.path import Path
import matplotlib.patches as patches
import scipy.io
import tensorflow as tf


flags = tf.app.flags
flags.DEFINE_string('data_dir', '/tmp/dat/', 'Directory for data')
flags.DEFINE_string('logdir', '/tmp/log/', 'Directory for logs')

flags.DEFINE_integer('latent_dim', 2, 'Latent dimensionality of model')
flags.DEFINE_integer('batch_size', 130, 'Minibatch size')
flags.DEFINE_integer('n_samples', 100, 'Number of samples to save')
flags.DEFINE_integer('print_every', 20, 'Print every n iterations')
flags.DEFINE_integer('hidden_size', 30, 'Hidden size for neural networks')
flags.DEFINE_integer('n_iterations', 1, 'number of iterations')
FLAGS = flags.FLAGS



Y =np.loadtxt('tsne_Y_values.txt')

labels = scipy.io.loadmat('/home/eunsongk/Data/ADNI/ADNI_eMCI_AAL116ROI_TimeSeries(0.01-0.08).mat')["labels"]

labels = np.array([labels] * 130).transpose()

labels = np.asarray(labels).ravel()

labels_tsne = np.asarray(labels)


n_samples = 8060

print(Y.shape)
print(labels_tsne.shape)


for i in range(FLAGS.n_iterations * (n_samples // FLAGS.batch_size)):
     offset = (i) % (n_samples // FLAGS.batch_size)
     # Re-binarize the data at every batch; this improves results
     # Original

     # np_x_fixed = arr2[offset * FLAGS.batch_size:(offset + 1) * FLAGS.batch_size].reshape(-1, input_dim)
     tsne_Y = Y[offset * FLAGS.batch_size:(offset + 1) * FLAGS.batch_size].reshape(-1, 2)
     tsne_Y_labels = labels_tsne[offset * FLAGS.batch_size:(offset + 1) * FLAGS.batch_size]

     if i in xrange((n_samples//FLAGS.batch_size)):
         fig = plt.figure(facecolor="white", figsize=(10.0, 8.0))
         plt.xlim(-150.0, 150.0)
         plt.ylim(-150.0, 150.0)
         plt.axis("off")


         cmap = plt.get_cmap('bwr')
         if tsne_Y_labels[0] == 1:
            plt.scatter(tsne_Y[:, 0], tsne_Y[:, 1], 20, c = tsne_Y_labels, cmap=mpl.colors.ListedColormap('red'))
         else:
            plt.scatter(tsne_Y[:, 0], tsne_Y[:, 1], 20, c= tsne_Y_labels, cmap=mpl.colors.ListedColormap('blue'))

         labels_plt = ['{0}'.format(j) for j in range(129)]

         #
         # for label, a, b in zip(labels_plt, tsne_Y[:, 0], tsne_Y[:, 1]):
         #     plt.annotate(label, xy=(a, b), xytext=(-0.07, 0.07),
         #                        textcoords='offset points', ha='right', va='bottom', arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
         #                 #
         # # plt.savefig(os.path.join(FLAGS.logdir, 'tSNE_map_frame_%d.png' % i))





         scat = plt.scatter([], [], c='white')

         def initiation():
              scat.set_offsets([])
              return scat,

         def animate(t):
             x_ani = tsne_Y[:, 0].transpose()
             y_ani = tsne_Y[:, 1].transpose()
             data_ani = np.hstack((x_ani[t:, np.newaxis], y_ani[t:, np.newaxis]))
             # print (data_ani)
             scat.set_offsets(data_ani)
             return scat,

#                 # ims = []
#                 # timepoint = []
#                 #
#                 # for a in scat():
#                 #     timepoint.append(a)
#                 #     ims.append(timepoint)

         ani = animation.FuncAnimation(fig, animate, init_func=initiation, frames=FLAGS.batch_size + 17 , interval=200, blit=True)
        # plt.show()

         Writer = animation.writers['ffmpeg']
         writer = Writer(fps=13, metadata=dict(artist='Kang, Eun Song (Korea University MiLab)'), bitrate=1800)


         ani.save(os.path.join(FLAGS.logdir, 'test_all_%d.mov'%i), writer=writer, dpi=300)