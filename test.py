import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print sess.run(hello)
# sess.close()

a = tf.constant(10)
b = tf.constant(32)
print sess.run(a+b)
sess.close()

for label, x, y in zip(labels, Y[:, 0], Y[:, 1]):
    plt.annotate(
        label, xy=(x, y), xytext=(-0.5, 0.5),
        textcoords='offset points', ha='right', va='bottom',
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

plt.savefig(os.path.join(FLAGS.logdir, 'tSNE_map_frame_%d.png' % i))
plt.close()
