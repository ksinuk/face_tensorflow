import tensorflow as tf

hello = tf.constant("hello world!")
sess = tf.Session()
print(sess.run(hello))

a = tf.constant(11)
b = tf.constant(21)
print(sess.run(a+b))

sess.close()